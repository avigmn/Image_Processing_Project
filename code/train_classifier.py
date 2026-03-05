import numpy as np
import os
import glob

def compute_mutual_information(feat_class1, feat_class2, num_bins=100):
    num_features = feat_class1.shape[1]
    best_thresholds = np.zeros(num_features)
    best_mi = np.zeros(num_features)
    
    prob_c1 = np.zeros(num_features)
    prob_c2 = np.zeros(num_features)
    
    n1 = feat_class1.shape[0]
    n2 = feat_class2.shape[0]
    n_total = n1 + n2
    
    for f in range(num_features):
        vals1 = np.abs(feat_class1[:, f])
        vals2 = np.abs(feat_class2[:, f])
        
        all_vals = np.concatenate((vals1, vals2))
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        
        thresholds = np.linspace(min_val, max_val, num_bins)
        max_mi_for_f = -1
        best_t_for_f = 0
        best_p1_for_f = 1e-6
        best_p2_for_f = 1e-6
        
        for t in thresholds:
            count1 = np.sum(vals1 >= t)
            count2 = np.sum(vals2 >= t)
            count_total = count1 + count2
            
            if count_total == 0 or count_total == n_total:
                continue 
                
            p_f = count_total / n_total
            p_f_given_c1 = count1 / n1
            p_f_given_c2 = count2 / n2
            
            mi_c1 = (p_f_given_c1 * np.log(p_f_given_c1 / p_f)) if p_f_given_c1 > 0 else 0
            mi_c2 = (p_f_given_c2 * np.log(p_f_given_c2 / p_f)) if p_f_given_c2 > 0 else 0
            
            current_mi = max(mi_c1, mi_c2)
            
            if current_mi > max_mi_for_f:
                max_mi_for_f = current_mi
                best_t_for_f = t
                # Save probabilities with tiny smoothing to prevent zero-division errors later
                best_p1_for_f = max(p_f_given_c1, 1e-6)
                best_p2_for_f = max(p_f_given_c2, 1e-6)
                
        best_thresholds[f] = best_t_for_f
        best_mi[f] = max_mi_for_f
        prob_c1[f] = best_p1_for_f
        prob_c2[f] = best_p2_for_f
        
    return best_thresholds, best_mi, prob_c1, prob_c2

if __name__ == "__main__":
    print("Loading extracted features for training...")
    data_dir = "../data/"
    
    waving_files = [f for f in glob.glob(os.path.join(data_dir, "waving_*_features.npy")) if "test" not in f.lower()]
    waving_features = np.vstack([np.load(f) for f in waving_files])
    
    walking_files = [f for f in glob.glob(os.path.join(data_dir, "walking_*_features.npy")) if "test" not in f.lower()]
    walking_features = np.vstack([np.load(f) for f in walking_files])
    
    print(f"Original: {waving_features.shape[0]} waving, {walking_features.shape[0]} walking.")
    
    sample_size = min(waving_features.shape[0], walking_features.shape[0])
    np.random.shuffle(waving_features)
    np.random.shuffle(walking_features)
    
    waving_sample = waving_features[:sample_size]
    walking_sample = walking_features[:sample_size]
    print(f"Balanced training set to {sample_size} samples per class.")
    
    print("Calculating MI and thresholds...")
    thresholds, mis, p_waving, p_walking = compute_mutual_information(waving_sample, walking_sample)
    
    # Ignore DC component (index 0). We want to classify based on movement, not average brightness
    mis[0] = 0 
    
    top_indices = np.argsort(mis)[-10:]
    
    print(f"Top feature indices selected: {top_indices}")
    
    model_params = {
        'indices': top_indices,
        'thresholds': thresholds[top_indices],
        'p_waving_given_f': p_waving[top_indices],
        'p_walking_given_f': p_walking[top_indices],
        'waving_prob': 0.5,
        'walking_prob': 0.5
    }
    np.save(os.path.join(data_dir, "naive_bayes_model.npy"), model_params, allow_pickle=True)
    print("Model saved.")
