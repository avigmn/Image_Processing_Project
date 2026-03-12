import numpy as np
import glob
import os
from scipy.fftpack import dctn

BLOCK_SIZE = 5
MOTION_THRESHOLD = 20.0


def extract_3d_dct_features(video_array, block_size=BLOCK_SIZE, motion_threshold=MOTION_THRESHOLD):
    time_frames, height, width = video_array.shape
    features = []
    margin = block_size // 2

    for t in range(margin, time_frames - margin, 2):
        for y in range(margin, height - margin, 2):
            for x in range(margin, width - margin, 2):
                block = video_array[t - margin: t + margin + 1,
                                    y - margin: y + margin + 1,
                                    x - margin: x + margin + 1].astype(np.float32)

                time_diff = np.abs(np.diff(block, axis=0))
                if np.mean(time_diff) < motion_threshold:
                    continue

                mean = block.mean()
                std = block.std()
                if std < 1e-6:
                    continue
                block_norm = (block - mean) / std

                features.append(dctn(block_norm, norm='ortho').flatten())

    return np.array(features)


if __name__ == "__main__":
    print("Starting 128x128 augmented (horizontal flip) feature extraction...")

    data_dir = "../data/"
    processed_videos = glob.glob(os.path.join(data_dir, "*_processed_128.npy"))

    if not processed_videos:
        print("No 128x128 processed arrays found. Run preprocess_128.py first.")

    for video_file in processed_videos:
        if "test" in video_file.lower():
            continue

        output_name = os.path.basename(video_file).replace("_processed_128.npy", "_augmented_128_features.npy")
        output_path = os.path.join(data_dir, output_name)

        if os.path.exists(output_path):
            print(f"Skipping {os.path.basename(video_file)} - augmented features already extracted.")
            continue

        print(f"Extracting flipped 128x128 features from {os.path.basename(video_file)}...")
        video_array = np.load(video_file)
        video_flipped = np.flip(video_array, axis=2)
        features = extract_3d_dct_features(video_flipped)
        np.save(output_path, features)
        print(f"Saved {features.shape[0]} augmented feature vectors to {output_path}")

    print("Augmented feature extraction complete.")
