#!/bin/bash
# Test the complete MLOps pipeline workflow

set -e

echo "=============================="
echo "MLOps Pipeline Workflow Test"
echo "=============================="

# Step 1: Generate data
echo ""
echo "Step 1: Generating synthetic data..."
python generate_raw_data.py --num-rows 500
echo "✓ Data generation completed"

# Step 2: Test data loading
echo ""
echo "Step 2: Testing data loading..."
python -c "
from src.data.data_loader import load_data_for_training
(X_train, y_train), (X_test, y_test), metadata = load_data_for_training('./data/raw/synthetic_data.csv')
print(f'✓ Train: {len(X_train)} samples')
print(f'✓ Test: {len(X_test)} samples')
print(f'✓ Features: {metadata[\"num_categorical_features\"]} cat + {metadata[\"num_numerical_features\"]} num')
"

# Step 3: Run pipeline examples
echo ""
echo "Step 3: Running pipeline examples..."
python pipeline_examples.py > /dev/null 2>&1
echo "✓ Pipeline examples completed"

# Step 4: Verify files
echo ""
echo "Step 4: Verifying generated files..."
[ -f data/raw/synthetic_data.csv ] && echo "✓ data/raw/synthetic_data.csv exists"
[ -f data/raw/synthetic_data_info.txt ] && echo "✓ data/raw/synthetic_data_info.txt exists"

# Summary
echo ""
echo "=============================="
echo "All tests passed! ✓"
echo "=============================="
echo ""
echo "Next steps:"
echo "  1. Install torch: pip install torch"
echo "  2. Train model: python train.py --train --epochs 5"
echo "  3. Start services: docker-compose up -d"
echo ""
