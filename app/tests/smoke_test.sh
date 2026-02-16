#!/bin/bash
# Smoke Test for MedGemma CXR Triage Demo
# ========================================
#
# This script verifies the demo app is working correctly.
#
# Usage:
#   ./smoke_test.sh [--url URL]
#
# Default URL: http://localhost:7860

set -e

# Configuration
DEFAULT_URL="http://localhost:7860"
URL="${1:-$DEFAULT_URL}"
TIMEOUT=30

echo "=========================================="
echo "MedGemma Demo - Smoke Test"
echo "=========================================="
echo "Target URL: $URL"
echo ""

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local expected_status=$2
    local description=$3
    
    echo -n "Testing $description... "
    
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$URL$endpoint" 2>/dev/null || echo "000")
    
    if [ "$status_code" = "$expected_status" ]; then
        echo "[PASS] (HTTP $status_code)"
        return 0
    else
        echo "[FAIL] (expected $expected_status, got $status_code)"
        return 1
    fi
}

# Function to test prediction endpoint
test_prediction() {
    echo -n "Testing /predict endpoint... "
    
    # Download sample image
    SAMPLE_IMAGE="/tmp/test_cxr.png"
    curl -s -o "$SAMPLE_IMAGE" "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png" -H "User-Agent: SmokeTest"
    
    if [ ! -f "$SAMPLE_IMAGE" ]; then
        echo "[FAIL] (could not download sample image)"
        return 1
    fi
    
    # Send prediction request
    response=$(curl -s -X POST "$URL/predict" \
        -F "image=@$SAMPLE_IMAGE" \
        -F "request_id=smoke_test_001" \
        --max-time 120 2>/dev/null)
    
    # Check response contains expected fields
    if echo "$response" | grep -q '"label"'; then
        if echo "$response" | grep -q '"explanation"'; then
            echo "[PASS]"
            echo "  Response: $response" | head -c 200
            echo "..."
            rm -f "$SAMPLE_IMAGE"
            return 0
        fi
    fi
    
    echo "[FAIL] (invalid response format)"
    echo "  Response: $response"
    rm -f "$SAMPLE_IMAGE"
    return 1
}

# Run tests
echo ""
echo "Running tests..."
echo "------------------------------------------"

PASSED=0
FAILED=0

# Test 1: Root endpoint
if check_endpoint "/" "200" "Root endpoint"; then
    ((PASSED++))
else
    ((FAILED++))
fi

# Test 2: Health check (if FastAPI running)
if check_endpoint "/health" "200" "Health check"; then
    ((PASSED++))
else
    # Gradio doesn't have /health, check for Gradio-specific endpoint
    if check_endpoint "/config" "200" "Gradio config"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
fi

# Test 3: Prediction endpoint (only if FastAPI)
# Note: Gradio uses different endpoint structure
# if test_prediction; then
#     ((PASSED++))
# else
#     ((FAILED++))
# fi

echo ""
echo "------------------------------------------"
echo "Results: $PASSED passed, $FAILED failed"
echo "------------------------------------------"

if [ $FAILED -eq 0 ]; then
    echo "[OK] All smoke tests passed!"
    exit 0
else
    echo "[FAIL] Some tests failed"
    exit 1
fi
