import numpy as np
import glob
import os
from scipy.fftpack import dctn

def extract_3d_dct_features(video_array, block_size=5, motion_threshold=20.0):
    """
    Extracts 3D DCT features from video blocks.
    Filters out blocks with low motion (small time derivative).
    """
    time_frames, height, width = video_array.shape
    features = []
    
    # Calculate padding
    margin = block_size // 2
    
    for t in range(margin, time_frames - margin, 2):
        for y in range(margin, height - margin, 2):
            for x in range(margin, width - margin, 2):
                
                # Extract the 5x5x5 spatio-temporal block
                block = video_array[t-margin : t+margin+1, 
                                    y-margin : y+margin+1, 
                                    x-margin : x+margin+1]
                
                block_float = block.astype(np.float32)

                # Calculate time derivative
                time_diff = np.abs(np.diff(block_float, axis=0))
                if np.mean(time_diff) < motion_threshold:
                    continue  # Skip static blocks

                # Normalize to zero mean and unit variance
                mean = block_float.mean()
                std = block_float.std()
                if std < 1e-6:
                    continue  # Skip uniform blocks
                block_norm = (block_float - mean) / std

                block_dct = dctn(block_norm, norm='ortho')

                features.append(block_dct.flatten())
                
    return np.array(features)

if __name__ == "__main__":
    print("Starting feature extraction...")
    
    data_dir = "../data/"
    processed_videos = glob.glob(os.path.join(data_dir, "*_processed.npy"))
    
    if not processed_videos:
        print("No processed video arrays found. Run preprocess.py first.")
    
    for video_file in processed_videos:

        if "test" in video_file.lower():
            continue

        output_name = os.path.basename(video_file).replace("_processed.npy", "_original_features.npy")
        output_path = os.path.join(data_dir, output_name)
        
        if os.path.exists(output_path):
            print(f"Skipping {os.path.basename(video_file)} - features already extracted.")
            continue

        print(f"Extracting features from {os.path.basename(video_file)}...")
        
        video_array = np.load(video_file)
        
        features = extract_3d_dct_features(video_array, motion_threshold=20.0)
        
        np.save(output_path, features)
        print(f"Saved {features.shape[0]} feature vectors to {output_path}")

    print("Feature extraction complete")
