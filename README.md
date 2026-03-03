# Movements Recognition using Spatio-Temporal Volumes

This project is a pure Python implementation of the action recognition algorithm presented in the 2005 paper: **"Efficient Visual Event Detection using Volumetric Features"** by Ke, Sukthankar, and Hebert. 

The system distinguishes between specific human actions (e.g., walking vs. waving) by extracting dense local spatio-temporal blocks, computing the 3D Discrete Cosine Transform (DCT), and classifying them using a Mutual Information-based Naive Bayes classifier.

## Directory Structure

* `code/`: Contains all Python scripts for the pipeline.
* `data/`: Contains input videos, the pre-trained model, and extracted numpy arrays.
* `results/`: Destination folder for the final annotated output videos.
* `run_experiment.py`: Master wrapper script for running the evaluation.
* `requirements.txt`: List of required Python packages.

## Prerequisites

Ensure you have Python 3 installed. Install the required dependencies using:
`pip install -r requirements.txt`

## How to Run the Experiment (Testing)

A pre-trained Naive Bayes model (`naive_bayes_model.npy`) is included in the `data/` folder for immediate evaluation. You do not need to retrain the model to test it.

1. Place your test video in the `data/` folder.
2. Ensure the filename includes the word **"test"** (e.g., `my_walking_test.mp4`).
3. From the main project directory, run the experiment script:
   `python run_experiment.py`
4. The system will process the video, draw bounding box annotations (green for walking, red for waving) over the detected local spatio-temporal blocks, and save the final video in the `results/` folder.

## Training Pipeline (Optional)

If you wish to retrain the model from scratch on your own dataset, follow these steps sequentially from within the `code/` directory:

1. **Preprocessing**: Converts raw `.mp4` training videos into 64x64 grayscale NumPy volumes. Automatically ignores files with "test" in the name.
   `python preprocess.py`
2. **Feature Extraction**: Slides a 5x5x5 window across the volumes, filters static blocks, and calculates the 3D DCT.
   `python extract_features.py`
3. **Training**: Computes Mutual Information to select the most discriminative DCT coefficients and saves the probability thresholds.
   `python train_classifier.py`
