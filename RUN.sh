#!/bin/bash
# Complete workflow script for Box Office Prediction

echo "======================================================================"
echo " BOX OFFICE PREDICTION - AUTOMATED SETUP & RUN"
echo "======================================================================"
echo ""

# Step 1: Create conda environment
echo "Step 1: Creating conda environment 'movie' with Python 3.11..."
conda create -n movie python=3.11 -y

# Step 2: Activate and install dependencies
echo ""
echo "Step 2: Installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate movie
pip install -r requirements.txt

# Step 3: Run pipeline
echo ""
echo "Step 3: Running ML pipeline..."
python main.py

# Step 4: Instructions
echo ""
echo "======================================================================"
echo " SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "To launch the Gradio web interface:"
echo "  conda activate movie"
echo "  python src/gradio_app.py"
echo ""
echo "Then open: http://localhost:7860"
echo "======================================================================"
