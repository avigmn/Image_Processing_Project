import cv2
import numpy as np
import os
import glob

def process_video(video_path, output_path):
    """
    Reads a video file, converts to grayscale, resizes to 64x64, 
    and saves as a NumPy array.
    """
    if not os.path.exists(video_path):
        print(f"Error: Could not find video at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to a resolution of 64x64
        resized = cv2.resize(gray, (64, 64))
        
        frames.append(resized)

    cap.release()
    
    # Convert to a 3D NumPy array (time, height, width)
    video_array = np.array(frames)
    
    np.save(output_path, video_array)
    print(f"Successfully processed {video_path}.")
    print(f"Output saved to {output_path} with shape: {video_array.shape}")

if __name__ == "__main__":
    print("Starting batch video preprocessing...")
    
    video_files = glob.glob("*.mp4")
    
    if not video_files:
        print("No .mp4 files found in the directory.")
    
    for input_video in video_files:
        output_data = input_video.replace(".mp4", "_processed.npy")
        
        if os.path.exists(output_data):
            print(f"Skipping {input_video} - already processed.")
            continue
            
        print(f"Processing: {input_video}...")
        process_video(input_video, output_data)
        
    print("Finished processing all videos")
