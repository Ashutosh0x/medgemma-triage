# Kaggle entry point
from app.services.agents import run_pipeline

if __name__ == "__main__":
    # Example usage for manual execution
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_pipeline(image_path=img_path)
    print(result)
