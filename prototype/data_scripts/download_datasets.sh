#!/bin/bash
# Dataset Download Instructions for MedGemma CXR Triage Assistant
# ================================================================

echo "=========================================="
echo "MedGemma Dataset Download Instructions"
echo "=========================================="

echo ""
echo "This script provides instructions for obtaining datasets."
echo "Due to data use agreements, datasets cannot be auto-downloaded."
echo ""

# MIMIC-CXR Instructions
echo "=========================================="
echo "1. MIMIC-CXR (PhysioNet)"
echo "=========================================="
echo ""
echo "MIMIC-CXR contains ~377,000 chest X-rays with radiology reports."
echo ""
echo "Steps to obtain access:"
echo "  1. Create a PhysioNet account: https://physionet.org/register/"
echo "  2. Complete CITI training (Data or Specimens Only Research)"
echo "  3. Submit credentialing application"
echo "  4. Wait for approval (typically 1-2 weeks)"
echo "  5. Sign data use agreement for MIMIC-CXR"
echo ""
echo "Dataset URLs:"
echo "  - MIMIC-CXR (DICOM): https://physionet.org/content/mimic-cxr/2.1.0/"
echo "  - MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/"
echo ""
echo "Download command (after approval):"
echo "  wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
echo ""

# CheXpert Instructions
echo "=========================================="
echo "2. CheXpert (Stanford AIMI)"
echo "=========================================="
echo ""
echo "CheXpert contains ~224,000 chest X-rays with uncertainty labels."
echo ""
echo "Steps to obtain access:"
echo "  1. Visit: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"
echo "  2. Create an account or sign in"
echo "  3. Accept the data use agreement"
echo "  4. Download the dataset"
echo ""

# Public Sample Images (CC0)
echo "=========================================="
echo "3. Public Sample Images (CC0 - for development)"
echo "=========================================="
echo ""
echo "For initial development, you can use CC0-licensed images:"
echo ""
echo "Sample CXR image (Wikimedia Commons):"
echo "  https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
echo ""

# Create sample data directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${SCRIPT_DIR}/../data"

echo "Creating data directories..."
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/processed"
mkdir -p "${DATA_DIR}/samples"

echo ""
echo "Data directories created:"
echo "  - ${DATA_DIR}/raw       (for raw dataset files)"
echo "  - ${DATA_DIR}/processed (for processed subsets)"
echo "  - ${DATA_DIR}/samples   (for sample images)"
echo ""

# Download sample CC0 image for development
echo "Downloading sample CC0 image for development..."
if command -v curl &> /dev/null; then
    curl -L -o "${DATA_DIR}/samples/sample_cxr_1.png" \
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png" \
        -H "User-Agent: MedGemmaProject"
    echo "Sample image downloaded to: ${DATA_DIR}/samples/sample_cxr_1.png"
else
    echo "curl not found. Please download manually."
fi

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Apply for dataset access (links above)"
echo "  2. Use sample images for initial development"
echo "  3. Run prepare_mimic_subset.py after obtaining MIMIC-CXR"
echo ""
