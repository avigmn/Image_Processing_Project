import os
import subprocess

def main():
    print("==================================================")
    print(" Movements Recognition based on the paper")
    print("==================================================")
    print("Running the pre-trained model on test videos...\n")
    
    if not os.path.exists("code"):
        print("Error: 'code' directory not found. Please run this script from the project root.")
        return

    subprocess.run(["python", "classify_video.py"], cwd="code")
    
    print("\n==================================================")
    print("Experiment finished! Please check the 'results/' folder.")
    print("==================================================")

if __name__ == "__main__":
    main()
