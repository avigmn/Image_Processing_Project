# Action Recognition using Spatio-Temporal DCT Features

A Python implementation of the activity recognition algorithm from the paper:
**"Recognizing image 'style' and activities in video using local features and naive Bayes"** by Daniel Keren (Pattern Recognition Letters, 2003).

The system classifies human actions (walking vs. waving) by extracting dense local spatio-temporal blocks, computing 3D Discrete Cosine Transform (DCT) features, and classifying them using Mutual Information-based feature selection. Two models are provided: a faithful reproduction of the original paper, and an improved ensemble model.

---

## Directory Structure

```
/code/          # Feature extraction, training, and classification scripts
/data/          # Input videos, pre-trained models, and preprocessing scripts
/results/       # Output annotated videos
/docs/          # Reports, slides, and AI assistance log
README.md
requirements.txt
run_experiment.py   # Run inference on test videos (interactive)
run_all.py          # Full pipeline from scratch (preprocess → train → classify)
```

---

## Models

### Original (Paper Faithful)
- Preprocessing: 64×64 grayscale frames
- Features: 5×5×5 spatio-temporal blocks → 3D DCT → 125-dim vectors
- Classifier: Naive Bayes with Mutual Information feature selection (top 10 DCT coefficients)
- Output: `results/*_original_classified.mp4`

### Ensemble (Improvement)
- Preprocessing: 128×128 grayscale frames (5.2× more training data)
- Data augmentation: horizontal flipping
- Features: same 3D DCT, top 20 MI-selected coefficients
- Classifiers: majority vote of 3 models — Naive Bayes + SVM (RBF) + Logistic Regression
- Post-processing: temporal smoothing, dominant-class suppression, blob cleanup
- Output: `results/*_ensemble_classified.mp4`

Colors used in both models: **yellow** = waving, **purple** = walking.

---

## Prerequisites

Python 3 is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run — Inference Only

Pre-trained models are included in `data/`. No retraining needed.

1. Place your test video in the `data/` folder. The filename must contain the word **"test"** (e.g., `my_test.mp4`).
2. From the project root, run:
   ```bash
   python run_experiment.py
   ```
3. You will be prompted to choose:
   ```
   Select model to run:
     1. Original (DCT + Naive Bayes)
     2. Ensemble (NB + SVM + LogReg)
     3. Both
   Enter choice [1/2/3]:
   ```
4. Annotated output videos are saved in the `results/` folder.

---

## Full Pipeline — Train from Scratch

To reproduce the full pipeline (preprocess → extract features → train → classify):

```bash
python run_all.py
```

This runs all steps sequentially and stops on any failure:

| Step | Script | Description |
|------|--------|-------------|
| 1a | `data/preprocess.py` | Convert training videos to 64×64 grayscale NumPy arrays |
| 1b | `data/preprocess_128.py` | Convert training videos to 128×128 grayscale NumPy arrays |
| 2a | `code/extract_features_original.py` | 3D DCT features from 64×64 frames |
| 2b | `code/extract_features_augmented.py` | Same + horizontal flip |
| 2c | `code/extract_features_128.py` | 3D DCT features from 128×128 frames |
| 2d | `code/extract_features_128_augmented.py` | Same + horizontal flip |
| 3a | `code/train_classifier_original.py` | Train Naive Bayes (64×64) |
| 3b | `code/train_classifier_dct_svm.py` | Train SVM (128×128) |
| 3c | `code/train_classifier_dct_logreg.py` | Train Logistic Regression (128×128) |
| 3d | `code/train_classifier_nb_128.py` | Train Naive Bayes (128×128) |
| 4a | `code/classify_video_original.py` | Classify with Original model |
| 4b | `code/classify_video_ensemble.py` | Classify with Ensemble model |

Training videos must be placed in `data/` and must **not** contain "test" in their filename.

---

## Improvements Over the Baseline

| # | Improvement | Effect |
|---|-------------|--------|
| 1 | Temporal smoothing + blob cleanup | Major visual improvement |
| 2 | SVM C=1 (cross-validated) | +2.7% accuracy |
| 3 | Data augmentation (horizontal flip) | Improved robustness |
| 4 | 128×128 resolution | 5.2× more training data — major improvement |
| 5 | Ensemble majority vote (NB + SVM + LogReg) | Improved robustness |
| 6 | Per-frame dominant-class suppression | Reduced cross-contamination |
