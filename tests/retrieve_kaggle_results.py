import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    print("Authenticating with Kaggle API...")
    # Ensure env var KAGGLE_API_TOKEN is set by the caller
    api = KaggleApi()
    api.authenticate()
    
    KERNEL_SLUG = "ashutosh0x/medgemma-high-fidelity-triage"
    DEST_DIR = "results_final_py"
    
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        
    print(f"Downloading output for {KERNEL_SLUG} to {DEST_DIR}...")
    try:
        api.kernels_output(KERNEL_SLUG, path=DEST_DIR)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        # Try to read the file if it was partially downloaded?
        # Or maybe it failed during 'print'?
        pass

if __name__ == "__main__":
    main()
