import cv2
import numpy as np
import os
import glob
import joblib
from scipy.fftpack import dctn

BLOCK_SIZE = 5          # spatial+temporal block size
MOTION_THRESHOLD = 20.0 # same as original paper
CONFIDENCE_THRESHOLD = 0.3  # SVM decision function margin
PROB_THRESHOLD = 0.55       # LogReg minimum predicted probability
NB_CONFIDENCE = 2.0         # NB winner must be >= this * loser (log-prob ratio)
MIN_BLOB_SIZE = 60          # minimum connected region size (pixels)
SMOOTH_WINDOW = 5           # number of consecutive frames for temporal majority-vote smoothing
DOMINANT_THRESHOLD = 0.45   # suppress minority class if its pixel ratio < this per frame


def extract_dct_block(block_float):
    """Normalize block and compute 3D DCT. Returns flattened 125-dim vector."""
    mean = block_float.mean()
    std  = block_float.std()
    if std < 1e-6:
        return None
    block_norm = (block_float - mean) / std
    return dctn(block_norm, norm='ortho').flatten()


def classify_video(video_path, nb_model_path, svm_model_path, logreg_model_path, output_path):
    print(f"\nLoading models...")
    nb_model  = np.load(nb_model_path, allow_pickle=True).item()
    svm_model = joblib.load(svm_model_path)
    lr_model  = joblib.load(logreg_model_path)

    # NB model parameters
    nb_indices    = nb_model['indices']
    nb_thresholds = nb_model['thresholds']
    nb_log_wave   = np.log(np.clip(nb_model['p_waving_given_f'],  1e-6, 1 - 1e-6))
    nb_log_nwave  = np.log(np.clip(1 - nb_model['p_waving_given_f'],  1e-6, 1 - 1e-6))
    nb_log_walk   = np.log(np.clip(nb_model['p_walking_given_f'], 1e-6, 1 - 1e-6))
    nb_log_nwalk  = np.log(np.clip(1 - nb_model['p_walking_given_f'], 1e-6, 1 - 1e-6))

    # SVM/LogReg model parameters
    svm         = svm_model['svm']
    svm_scaler  = svm_model['scaler']
    svm_indices = svm_model['indices']
    clf         = lr_model['clf']
    lr_scaler   = lr_model['scaler']
    lr_indices  = lr_model['indices']

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = int(cap.get(cv2.CAP_PROP_FPS))

    frames_color   = []
    frames_gray128 = []

    print("Reading frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_color.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray128.append(cv2.resize(gray, (128, 128)))
    cap.release()

    video_128   = np.array(frames_gray128, dtype=np.float32)
    T, H, W     = video_128.shape
    frame_count = T

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    margin   = BLOCK_SIZE // 2
    svm_maps = np.zeros((frame_count, H, W), dtype=np.int8)
    lr_maps  = np.zeros((frame_count, H, W), dtype=np.int8)
    nb_maps  = np.zeros((frame_count, H, W), dtype=np.int8)

    print("Classifying pixels (all three models at 128x128)...")
    for t in range(margin, T - margin):
        block_positions = []
        dct_vecs        = []

        for y in range(margin, H - margin):
            for x in range(margin, W - margin):
                block = video_128[t - margin: t + margin + 1,
                                  y - margin: y + margin + 1,
                                  x - margin: x + margin + 1]
                time_diff = np.abs(np.diff(block, axis=0))
                if np.mean(time_diff) < MOTION_THRESHOLD:
                    continue
                dct_vec = extract_dct_block(block)
                if dct_vec is None:
                    continue
                dct_vecs.append(dct_vec)
                block_positions.append((y, x))

        if block_positions:
            dct_arr = np.array(dct_vecs)   # (N, 125)

            # --- SVM (raw signed DCT) ---
            X_svm    = svm_scaler.transform(dct_arr[:, svm_indices])
            svm_conf = svm.decision_function(X_svm)
            svm_pred = svm.predict(X_svm)
            for (y, x), pred, conf in zip(block_positions, svm_pred, svm_conf):
                if abs(conf) >= CONFIDENCE_THRESHOLD:
                    svm_maps[t, y, x] = pred

            # --- LogReg (abs DCT) ---
            X_lr    = lr_scaler.transform(np.abs(dct_arr[:, lr_indices]))
            probas  = clf.predict_proba(X_lr)
            lr_pred = clf.predict(X_lr)
            for (y, x), pred, prob in zip(block_positions, lr_pred, probas):
                if prob.max() >= PROB_THRESHOLD:
                    lr_maps[t, y, x] = pred

            # --- NB 128x128 (vectorized, abs DCT) ---
            nb_feats = np.abs(dct_arr[:, nb_indices])
            above    = nb_feats >= nb_thresholds
            lw  = np.where(above, nb_log_wave,  nb_log_nwave).sum(axis=1) + np.log(0.5)
            lwk = np.where(above, nb_log_walk,  nb_log_nwalk).sum(axis=1) + np.log(0.5)
            winner = np.maximum(lw, lwk)
            loser  = np.minimum(lw, lwk)
            confident = winner >= np.log(NB_CONFIDENCE) + loser
            nb_pred = np.where(lw > lwk, 1, 2).astype(np.int8)
            for idx, (y, x) in enumerate(block_positions):
                if confident[idx]:
                    nb_maps[t, y, x] = nb_pred[idx]

        if t % 10 == 0:
            print(f"  Frame {t}/{T - margin - 1}...")

    # Ensemble: majority vote — label a pixel if at least 2 of 3 models agree
    print("Computing ensemble (majority vote: 2 of 3 models)...")
    label_maps = np.zeros((frame_count, H, W), dtype=np.int8)
    for label in [1, 2]:
        votes = ((svm_maps == label).astype(np.int8) +
                 (lr_maps  == label).astype(np.int8) +
                 (nb_maps  == label).astype(np.int8))
        label_maps[votes >= 2] = label

    # Temporal smoothing
    print("Applying temporal smoothing...")
    smoothed_maps = np.zeros_like(label_maps)
    half_s = SMOOTH_WINDOW // 2
    for t in range(frame_count):
        t0 = max(0, t - half_s)
        t1 = min(frame_count, t + half_s + 1)
        window = label_maps[t0:t1]
        waving_votes  = (window == 1).sum(axis=0)
        walking_votes = (window == 2).sum(axis=0)
        smoothed_maps[t][waving_votes  > walking_votes] = 1
        smoothed_maps[t][walking_votes > waving_votes]  = 2

    # Per-frame dominant-class suppression
    print("Applying dominant-class suppression...")
    for t in range(frame_count):
        waving_px  = np.sum(smoothed_maps[t] == 1)
        walking_px = np.sum(smoothed_maps[t] == 2)
        total_px   = waving_px + walking_px
        if total_px == 0:
            continue
        waving_ratio = waving_px / total_px
        if waving_ratio < DOMINANT_THRESHOLD:
            smoothed_maps[t][smoothed_maps[t] == 1] = 0
        elif waving_ratio > (1 - DOMINANT_THRESHOLD):
            smoothed_maps[t][smoothed_maps[t] == 2] = 0

    print("Drawing per-pixel results...")
    for i in range(frame_count):
        frame = frames_color[i].copy()
        lm_scaled = cv2.resize(
            smoothed_maps[i].astype(np.uint8),
            (orig_width, orig_height),
            interpolation=cv2.INTER_NEAREST
        )

        lm_clean = np.zeros_like(lm_scaled)
        for label in [1, 2]:
            mask = (lm_scaled == label).astype(np.uint8)
            n_labels, cc_map, stats, _ = cv2.connectedComponentsWithStats(mask)
            for comp in range(1, n_labels):
                if stats[comp, cv2.CC_STAT_AREA] >= MIN_BLOB_SIZE:
                    lm_clean[cc_map == comp] = label

        frame[lm_clean == 1] = (0, 255, 255)   # Yellow for waving
        frame[lm_clean == 2] = (128, 0, 128)   # Purple for walking

        cv2.putText(frame, "Model: Ensemble (NB + SVM + LogReg)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)

    out.release()
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    nb_model_file     = "../data/nb_128_model.npy"
    svm_model_file    = "../data/dct_svm_model.joblib"
    logreg_model_file = "../data/dct_logreg_model.joblib"
    os.makedirs("../results", exist_ok=True)

    for f, name in [(nb_model_file, "NB 128x128"), (svm_model_file, "DCT+SVM"), (logreg_model_file, "DCT+LogReg")]:
        if not os.path.exists(f):
            print(f"{name} model not found: {f}")
            exit(1)

    print("Searching for test videos...")
    test_videos = glob.glob("../data/*test*.mp4")

    if not test_videos:
        print("No test videos found.")
    else:
        for input_vid in test_videos:
            base_name   = os.path.basename(input_vid)
            output_name = base_name.replace(".mp4", "_ensemble_classified.mp4")
            output_vid  = os.path.join("../results", output_name)
            classify_video(input_vid, nb_model_file, svm_model_file, logreg_model_file, output_vid)
