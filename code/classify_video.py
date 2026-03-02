import cv2
import numpy as np
import os
from scipy.fftpack import dctn

def classify_video(video_path, model_path, output_path, block_size=5, motion_threshold=30.0):
    print(f"\nLoading model from {model_path}...")
    model = np.load(model_path, allow_pickle=True).item()
    
    indices = model['indices']
    thresholds = model['thresholds']
    p_wave_given_f = model['p_waving_given_f']
    p_walk_given_f = model['p_walking_given_f']
    
    # Cap probabilities to prevent log(0) issues
    p_wave_given_f = np.clip(p_wave_given_f, 1e-6, 1.0 - 1e-6)
    p_walk_given_f = np.clip(p_walk_given_f, 1e-6, 1.0 - 1e-6)
    
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_color = []
    frames_gray_64 = []
    
    print("Reading frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_color.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        frames_gray_64.append(resized)
        
    cap.release()
    
    video_array = np.array(frames_gray_64, dtype=np.float32)
    time_frames, height, width = video_array.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
    
    scale_x = orig_width / 64.0
    scale_y = orig_height / 64.0
    margin = block_size // 2
    
    boxes_per_frame = {i: [] for i in range(frame_count)}
    
    print("Classifying local blocks (Paper Implementation)...")
    for t in range(margin, time_frames - margin, 2):
        for y in range(margin, height - margin, 2):
            for x in range(margin, width - margin, 2):
                
                block = video_array[t-margin : t+margin+1, 
                                    y-margin : y+margin+1, 
                                    x-margin : x+margin+1]
                
                time_diff = np.abs(np.diff(block, axis=0))
                if np.mean(time_diff) < motion_threshold:
                    continue  # Skip on backgrounds 
                
                block_dct = dctn(block, norm='ortho').flatten()
                selected_features = np.abs(block_dct[indices])
                
                log_p_wave = np.log(0.5)
                log_p_walk = np.log(0.5)
                
                for i in range(len(indices)):
                    if selected_features[i] >= thresholds[i]:
                        log_p_wave += np.log(p_wave_given_f[i])
                        log_p_walk += np.log(p_walk_given_f[i])
                    else:
                        log_p_wave += np.log(1.0 - p_wave_given_f[i])
                        log_p_walk += np.log(1.0 - p_walk_given_f[i])
                
                label = 'waving' if log_p_wave > log_p_walk else 'walking'
                
                x1 = int((x - margin) * scale_x)
                y1 = int((y - margin) * scale_y)
                x2 = int((x + margin + 1) * scale_x)
                y2 = int((y + margin + 1) * scale_y)
                
                boxes_per_frame[t].append((x1, y1, x2, y2, label))

    print("Drawing local block results...")
    for i in range(frame_count):
        frame = frames_color[i].copy()
        
        for box in boxes_per_frame[i]:
            x1, y1, x2, y2, label = box
            if label == 'waving':
                color = (0, 0, 255) # Red for waving
            else:
                color = (0, 255, 0) # Green for walking
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
        out.write(frame)
        
    out.release()
    print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    model_file = "../data/naive_bayes_model.npy"
    os.makedirs("../results", exist_ok=True)
    
    test_videos = [
        ("../data/walking_test.mp4", "../results/walking_test_classified.mp4"),
        ("../data/waving_test.mp4", "../results/waving_test_classified.mp4")
    ]
    
    for input_vid, output_vid in test_videos:
        if os.path.exists(input_vid):
            classify_video(input_vid, model_file, output_vid, motion_threshold=20.0)
        else:
            print(f"\nCould not find {input_vid}. Skipping.")
