import numpy as np
import os
import glob

TOP_N_FEATURES = 20  # match SVM/LogReg feature count


def compute_mutual_information(feat_class1, feat_class2, num_bins=100):
    num_features = feat_class1.shape[1]
    best_thresholds = np.zeros(num_features)
    best_mi         = np.zeros(num_features)
    prob_c1         = np.zeros(num_features)
    prob_c2         = np.zeros(num_features)

    n1      = feat_class1.shape[0]
    n2      = feat_class2.shape[0]
    n_total = n1 + n2

    for f in range(num_features):
        vals1    = np.abs(feat_class1[:, f])
        vals2    = np.abs(feat_class2[:, f])
        all_vals = np.concatenate((vals1, vals2))
        thresholds = np.linspace(all_vals.min(), all_vals.max(), num_bins)

        max_mi_for_f  = -1
        best_t_for_f  = 0
        best_p1_for_f = 1e-6
        best_p2_for_f = 1e-6

        for t in thresholds:
            count1 = np.sum(vals1 >= t)
            count2 = np.sum(vals2 >= t)
            count_total = count1 + count2
            if count_total == 0 or count_total == n_total:
                continue
            p_f     = count_total / n_total
            p_f_c1  = count1 / n1
            p_f_c2  = count2 / n2
            mi_c1   = p_f_c1 * np.log(p_f_c1 / p_f) if p_f_c1 > 0 else 0
            mi_c2   = p_f_c2 * np.log(p_f_c2 / p_f) if p_f_c2 > 0 else 0
            current_mi = max(mi_c1, mi_c2)
            if current_mi > max_mi_for_f:
                max_mi_for_f  = current_mi
                best_t_for_f  = t
                best_p1_for_f = max(p_f_c1, 1e-6)
                best_p2_for_f = max(p_f_c2, 1e-6)

        best_thresholds[f] = best_t_for_f
        best_mi[f]         = max_mi_for_f
        prob_c1[f]         = best_p1_for_f
        prob_c2[f]         = best_p2_for_f

    return best_thresholds, best_mi, prob_c1, prob_c2


if __name__ == "__main__":
    print("Loading 128x128 DCT features for Naive Bayes training...")
    data_dir = "../data/"

    waving_files = [
        f for f in glob.glob(os.path.join(data_dir, "waving_*_original_128_features.npy"))
        if "test" not in f.lower()
    ]
    walking_files = [
        f for f in glob.glob(os.path.join(data_dir, "walking_*_original_128_features.npy"))
        if "test" not in f.lower()
    ]

    if not waving_files or not walking_files:
        print("128x128 feature files not found. Run extract_features_128.py first.")
        exit(1)

    waving_features  = np.vstack([np.load(f) for f in waving_files])
    walking_features = np.vstack([np.load(f) for f in walking_files])
    print(f"Original: {waving_features.shape[0]} waving, {walking_features.shape[0]} walking.")

    # Load augmented features if available
    waving_aug  = [f for f in glob.glob(os.path.join(data_dir, "waving_*_augmented_128_features.npy"))  if "test" not in f.lower()]
    walking_aug = [f for f in glob.glob(os.path.join(data_dir, "walking_*_augmented_128_features.npy")) if "test" not in f.lower()]
    if waving_aug and walking_aug:
        waving_features  = np.vstack([waving_features]  + [np.load(f) for f in waving_aug])
        walking_features = np.vstack([walking_features] + [np.load(f) for f in walking_aug])
        print(f"After augmentation: {waving_features.shape[0]} waving, {walking_features.shape[0]} walking.")

    # Balance classes
    sample_size = min(waving_features.shape[0], walking_features.shape[0])
    rng = np.random.default_rng(seed=42)
    waving_sample  = waving_features[rng.permutation(waving_features.shape[0])[:sample_size]]
    walking_sample = walking_features[rng.permutation(walking_features.shape[0])[:sample_size]]
    print(f"Balanced training set: {sample_size} samples per class.")

    print("Computing MI and thresholds...")
    thresholds, mis, p_waving, p_walking = compute_mutual_information(waving_sample, walking_sample)
    mis[0] = 0  # exclude DC component

    top_indices = np.argsort(mis)[-TOP_N_FEATURES:]
    print(f"Top {TOP_N_FEATURES} feature indices: {sorted(top_indices)}")

    model_params = {
        'indices':           top_indices,
        'thresholds':        thresholds[top_indices],
        'p_waving_given_f':  p_waving[top_indices],
        'p_walking_given_f': p_walking[top_indices],
    }
    model_path = os.path.join(data_dir, "nb_128_model.npy")
    np.save(model_path, model_params, allow_pickle=True)
    print(f"Model saved to {model_path}")
