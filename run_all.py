import os
import subprocess
import sys


def run(script, cwd=None, label=""):
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], cwd=cwd)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with return code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(root, "code")
    data_dir = os.path.join(root, "data")

    print("\n" + "="*50)
    print(" Full Pipeline — Original + Ensemble")
    print("="*50)
    print("Steps: preprocess -> extract features -> train -> classify")

    # Step 1: Preprocess videos (64x64 for original NB, 128x128 for ensemble)
    run("preprocess.py",     cwd=data_dir, label="Step 1a — Preprocessing videos (64x64)")
    run("preprocess_128.py", cwd=data_dir, label="Step 1b — Preprocessing videos (128x128)")

    # Step 2: Extract DCT features
    run("extract_features_original.py",    cwd=code_dir, label="Step 2a — Extracting DCT features (64x64, original)")
    run("extract_features_augmented.py",   cwd=code_dir, label="Step 2b — Extracting DCT features (64x64, flipped)")
    run("extract_features_128.py",         cwd=code_dir, label="Step 2c — Extracting DCT features (128x128, original)")
    run("extract_features_128_augmented.py", cwd=code_dir, label="Step 2d — Extracting DCT features (128x128, flipped)")

    # Step 3: Train all models
    run("train_classifier_original.py",   cwd=code_dir, label="Step 3a — Training Original Naive Bayes (64x64)")
    run("train_classifier_dct_svm.py",    cwd=code_dir, label="Step 3b — Training DCT + SVM (128x128, for ensemble)")
    run("train_classifier_dct_logreg.py", cwd=code_dir, label="Step 3c — Training DCT + LogReg (128x128, for ensemble)")
    run("train_classifier_nb_128.py",     cwd=code_dir, label="Step 3d — Training NB 128x128 (for ensemble)")

    # Step 4: Classify test videos
    run("classify_video_original.py",  cwd=code_dir, label="Step 4a — Classifying with Original (DCT + Naive Bayes)")
    run("classify_video_ensemble.py",  cwd=code_dir, label="Step 4b — Classifying with Ensemble (NB + SVM + LogReg)")

    print("\n" + "="*50)
    print(" All done! Check the results/ folder.")
    print("="*50)
