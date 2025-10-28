#!/bin/bash

echo "Cleaning up unused files..."

# Remove old inference scripts (keeping only production one)
rm -f inference.py inference_rf.py inference_best.py inference_mock.py test_model.py
echo "✓ Removed old inference scripts"

# Remove old MDN model files
rm -f datasets/ml_preprocessed/models/mdn_*.pt
rm -f datasets/ml_preprocessed/models/scaler_*.pth
rm -f datasets/ml_preprocessed/models/best_model_nn.pt
rm -f datasets/ml_preprocessed/models/model_metadata.pkl
echo "✓ Removed old neural network models"

# Remove old sklearn models (keeping production ones)
rm -f datasets/ml_preprocessed/models/redshift_model.pkl
rm -f datasets/ml_preprocessed/models/scaler_x.pkl
rm -f datasets/ml_preprocessed/models/scaler_y.pkl
echo "✓ Removed old sklearn models"

# Move training scripts to a scripts directory
mkdir -p scripts
mv train_best_model.py export_gb_model.py prepare_training_data.py scripts/ 2>/dev/null
mv train_pipeline.sh scripts/ 2>/dev/null
echo "✓ Organized training scripts"

# Archive old Python scripts
mkdir -p archive
mv analyze_features.py build_ml_tensors.py catalogs.py model-1.py archive/ 2>/dev/null
echo "✓ Archived old scripts"

echo ""
echo "Cleanup complete! Keeping:"
echo "  - inference_production.py (active inference)"
echo "  - scripts/ (training pipeline)"
echo "  - datasets/ml_preprocessed/models/production_*.pkl (active models)"
echo "  - backend/ and frontend/"

