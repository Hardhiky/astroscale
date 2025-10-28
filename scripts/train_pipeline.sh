#!/bin/bash

set -e  # Exit on error

echo "================================================================================"
echo "STELLAR REDSHIFT PREDICTION - COMPLETE TRAINING PIPELINE"
echo "================================================================================"

cd "$(dirname "$0")"

# Step 1: Prepare training data
echo ""
echo "Step 1/3: Preparing training data..."
echo "--------------------------------------------------------------------------------"
python3 prepare_training_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to prepare training data"
    exit 1
fi

# Check if training data was created
if [ ! -f "datasets/ml_preprocessed/stellar_redshift_training.csv" ]; then
    echo "❌ Training data file not created"
    exit 1
fi

echo ""
echo "✓ Training data prepared successfully"

# Step 2: Train models
echo ""
echo "Step 2/3: Training models (this may take several minutes)..."
echo "--------------------------------------------------------------------------------"
python3 train_best_model.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to train models"
    exit 1
fi

echo ""
echo "✓ Models trained successfully"

# Step 3: Update backend and rebuild
echo ""
echo "Step 3/3: Rebuilding backend..."
echo "--------------------------------------------------------------------------------"
cd backend
dune build

if [ $? -ne 0 ]; then
    echo "❌ Failed to build backend"
    exit 1
fi

echo ""
echo "✓ Backend rebuilt successfully"

# Summary
echo ""
echo "================================================================================"
echo "TRAINING PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "✓ Training data created"
echo "✓ Best model trained and saved"
echo "✓ Backend ready to use"
echo ""
echo "To start the services:"
echo "  Terminal 1: cd backend && dune exec backend"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Models saved in: datasets/ml_preprocessed/models/"
echo "================================================================================"
