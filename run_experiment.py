"""
Run inference on all videos with 'test' in their filename (in data/).
"""

import os
import subprocess
import sys


def run(script, label):
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], cwd="code")
    if result.returncode != 0:
        print(f"ERROR: {script} failed with return code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    if not os.path.exists("code"):
        print("Error: 'code' directory not found. Please run this script from the project root.")
        sys.exit(1)

    print("\n" + "="*50)
    print(" Action Recognition — Inference")
    print("="*50)
    print("\nSelect model to run:")
    print("  1. Original (DCT + Naive Bayes)")
    print("  2. Ensemble (NB + SVM + LogReg)")
    print("  3. Both")

    while True:
        choice = input("\nEnter choice [1/2/3]: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    if choice in ("1", "3"):
        run("classify_video_original.py", "Original model (DCT + Naive Bayes)")

    if choice in ("2", "3"):
        run("classify_video_ensemble.py", "Ensemble model (NB + SVM + LogReg)")

    print("\n" + "="*50)
    print(" Done! Check the results/ folder.")
    print("="*50)


if __name__ == "__main__":
    main()
