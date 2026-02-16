import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("# MedGemma High Fidelity Clinical Triage\n\nThis notebook demonstrates the multi-agent clinical triage pipeline built on Google MedGemma 1.5."))

cells.append(nbf.v4.new_code_cell(
"""# Clone the submission repo (explicitly main branch)
!git clone -b main https://github.com/Ashutosh0x/medgemma-triage.git
%cd medgemma-triage

# Diagnostic check for encoding issues
import os
with open('app/services/agents.py', 'rb') as f:
    content = f.read(100)
    print(f"File check: {content[:10].hex()}")
    if b'\\x00' in content:
        print("CRITICAL: Null bytes detected in agents.py!")

!pip install -r requirements.txt"""
))

cells.append(nbf.v4.new_code_cell(
"""import os
import torch
from app.services.agents import run_pipeline

# Configure model path for Kaggle
os.environ["MEDGEMMA_MODEL_ID"] = "/kaggle/input/medgemma/transformers/medgemma-1.5-4b-it/1"

print("Pipeline ready! MedGemma 1.5 loaded from Kaggle Models.")"""
))

cells.append(nbf.v4.new_code_cell(
"""# Demo Inference on a sample image from the dataset
sample_image = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"

if os.path.exists(sample_image):
    result = run_pipeline(image_path=sample_image)
    print("--- INFERENCE RESULT ---")
    print(f"Decision: {result.get('label')}")
    print(f"Confidence: {result.get('model_confidence')}")
    print(f"Explanation: {result.get('explanation')}")
else:
    print("Sample image not found. Please verify dataset mount.")"""
))

cells.append(nbf.v4.new_markdown_cell("## Visual Output Artifact\n\nBelow is the generated triage overlay produced by the MedGemma pipeline."))

cells.append(nbf.v4.new_code_cell(
"""from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import textwrap

if os.path.exists(sample_image) and result:
    img = Image.open(sample_image).convert("RGB")

    # Prepare overlay text from pipeline output
    triage = result.get("label", "UNKNOWN")
    confidence = result.get("model_confidence", "N/A")
    explanation = result.get("explanation", "")[:100] + "..."

    overlay_text = f"TRIAGE: {triage}\\nCONF: {confidence}\\n{explanation}"

    draw = ImageDraw.Draw(img)

    # Wrap text nicely
    wrapped = "\\n".join(textwrap.wrap(overlay_text, width=40))
    
    # Draw simple overlay box
    draw.rectangle([(0,0),(img.width,200)], fill=(0,0,0))
    draw.text((20,20), wrapped, fill=(255,255,255))

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("MedGemma Clinical Triage Output")
    plt.show()

    img.save("triage_output.png")
    print("Visual artifact saved: triage_output.png")
"""
))

nb["cells"] = cells

# Add kernelspec metadata
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.12"
    }
}

nbf.write(nb, "kaggle-notebook/notebook.ipynb")
print("Notebook generated successfully.")
