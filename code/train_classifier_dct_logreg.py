import numpy as np
import os
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

TOP_N_FEATURES = 20  # number of DCT coefficients to select by mutual information


def compute_mi_scores(feat_class1, feat_class2, num_bins=100):
    """
    Compute mutual information score per feature.
    Returns: MI score array (one value per feature)
    """
    num_features = feat_class1.shape[1]
    best_mi = np.zeros(num_features)
    n1, n2 = feat_class1.shape[0], feat_class2.shape[0]
    n_total = n1 + n2

    for f in range(num_features):
        vals1 = np.abs(feat_class1[:, f])
        vals2 = np.abs(feat_class2[:, f])
        all_vals = np.concatenate((vals1, vals2))
        min_val, max_val = all_vals.min(), all_vals.max()
        thresholds = np.linspace(min_val, max_val, num_bins)
        max_mi = -1

        for t in thresholds:
            count1 = np.sum(vals1 >= t)
            count2 = np.sum(vals2 >= t)
            count_total = count1 + count2
            if count_total == 0 or count_total == n_total:
                continue
            p_f = count_total / n_total
            p_f_c1 = count1 / n1
            p_f_c2 = count2 / n2
            mi_c1 = p_f_c1 * np.log(p_f_c1 / p_f) if p_f_c1 > 0 else 0
            mi_c2 = p_f_c2 * np.log(p_f_c2 / p_f) if p_f_c2 > 0 else 0
            mi = max(mi_c1, mi_c2)
            if mi > max_mi:
                max_mi = mi

        best_mi[f] = max_mi

    return best_mi


if __name__ == "__main__":
    print("Loading DCT features for DCT + Logistic Regression training...")

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
        print("128x128 DCT feature files not found. Run extract_features_128.py first.")
        exit(1)

    waving_features = np.vstack([np.load(f) for f in waving_files])
    walking_features = np.vstack([np.load(f) for f in walking_files])
    print(f"Original: {waving_features.shape[0]} waving, {walking_features.shape[0]} walking.")

    # Load augmented (horizontally flipped) features if available
    waving_aug = [f for f in glob.glob(os.path.join(data_dir, "waving_*_augmented_128_features.npy")) if "test" not in f.lower()]
    walking_aug = [f for f in glob.glob(os.path.join(data_dir, "walking_*_augmented_128_features.npy")) if "test" not in f.lower()]
    if waving_aug and walking_aug:
        waving_features = np.vstack([waving_features] + [np.load(f) for f in waving_aug])
        walking_features = np.vstack([walking_features] + [np.load(f) for f in walking_aug])
        print(f"After augmentation: {waving_features.shape[0]} waving, {walking_features.shape[0]} walking.")

    # Balance classes
    sample_size = min(waving_features.shape[0], walking_features.shape[0])
    rng = np.random.default_rng(seed=42)
    waving_sample = waving_features[rng.permutation(waving_features.shape[0])[:sample_size]]
    walking_sample = walking_features[rng.permutation(walking_features.shape[0])[:sample_size]]
    print(f"Balanced training set: {sample_size} samples per class.")

    # MI-based feature selection (exclude DC component at index 0)
    print("Computing mutual information for feature selection...")
    mi_scores = compute_mi_scores(waving_sample, walking_sample)
    mi_scores[0] = 0  # exclude DC component
    top_indices = np.argsort(mi_scores)[-TOP_N_FEATURES:]
    print(f"Top {TOP_N_FEATURES} DCT feature indices: {sorted(top_indices)}")

    # Build feature matrix using absolute DCT values (MI selection uses abs; sign is arbitrary)
    X_waving  = np.abs(waving_sample[:, top_indices])
    X_walking = np.abs(walking_sample[:, top_indices])
    X = np.vstack([X_waving, X_walking])
    y = np.array([1] * sample_size + [2] * sample_size)  # 1=waving, 2=walking

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Logistic Regression (C=1.0, L2)...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    print("Training complete.")

    model = {'clf': clf, 'scaler': scaler, 'indices': top_indices}
    model_path = os.path.join(data_dir, "dct_logreg_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
