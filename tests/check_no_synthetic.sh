#!/bin/bash
# Check for Synthetic/Mock Data
# ==============================
#
# CI script that fails if synthetic or mock data is detected
# in dataset directories or manifests.
#
# Usage:
#   ./check_no_synthetic.sh
#
# Exit codes:
#   0 - No synthetic data found
#   1 - Synthetic data detected

set -e

echo "=========================================="
echo "Checking for Synthetic/Mock Data"
echo "=========================================="

ERRORS=0

# Patterns that indicate synthetic/mock data
PATTERNS=(
    "synthetic"
    "fake"
    "mock"
    "simulated"
    "generated"
    "artificial"
    "demo_sample"
    "test_fake"
)

# Directories to check
CHECK_DIRS=(
    "prototype/data"
    "demo_app/data"
    "prototype/eval"
)

# Check file names
echo ""
echo "Checking file names for suspicious patterns..."
for dir in "${CHECK_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        for pattern in "${PATTERNS[@]}"; do
            matches=$(find "$dir" -iname "*${pattern}*" 2>/dev/null || true)
            if [ -n "$matches" ]; then
                echo "[FAIL] FOUND: Files matching '$pattern' in $dir:"
                echo "$matches" | head -5
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi
done

# Check manifest files for warnings
echo ""
echo "Checking manifest files..."
for manifest in $(find . -name "*manifest*.json" 2>/dev/null); do
    if grep -q '"warnings":\s*\[' "$manifest"; then
        warning_count=$(grep -c '"SUSPICIOUS' "$manifest" 2>/dev/null || echo "0")
        if [ "$warning_count" -gt "0" ]; then
            echo "[WARN] Manifest $manifest has $warning_count warnings"
        fi
    fi
    
    # Check for unknown sources
    if grep -q '"source":\s*"unknown"' "$manifest"; then
        echo "[FAIL] FOUND: Manifest $manifest has unknown source"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for mock mode in evaluation outputs
echo ""
echo "Checking evaluation outputs..."
for eval_file in $(find . -path "*/eval/*.json" 2>/dev/null); do
    if grep -qi "mock\|demo\|simulated" "$eval_file"; then
        echo "[WARN] WARNING: $eval_file may contain mock data references"
    fi
done

# Final report
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "[OK] No synthetic/mock data detected"
    exit 0
else
    echo "[FAIL] FAILED: Found $ERRORS issues"
    echo ""
    echo "Please:"
    echo "1. Remove synthetic/mock files from data directories"
    echo "2. Update manifests with proper source attribution"
    echo "3. Ensure evaluation uses only real datasets"
    exit 1
fi
