# Box Office Prediction - Quick Start Guide

Complete ML pipeline for predicting movie box office revenue from IMDb data.

## 🚀 Quick Start

### 1. Create Conda Environment
```bash
conda create -n movie python=3.11 -y
conda activate movie
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Pipeline
```bash
python main.py
```

This will:
- Clean and preprocess the data
- Train a Random Forest model
- Generate EDA visualizations
- Save the trained model

### 4. Launch Interactive Gradio App
```bash
python src/gradio_app.py
```

Then open your browser to `http://localhost:7860`

## 📋 Step-by-Step Instructions

### Setup (One-time)
```bash
# 1. Create and activate conda environment
conda create -n movie python=3.11 -y
conda activate movie

# 2. Install required packages
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Make sure you're in the project directory
cd /workspace/bor-prediction-analysis

# Activate environment
conda activate movie

# Run the complete pipeline
python main.py
```

### Launch Web Interface
```bash
# After pipeline completes, launch Gradio
conda activate movie
python src/gradio_app.py
```

Access the app at: **http://localhost:7860**

## 📁 Project Structure

```
bor-prediction-analysis/
├── main.py                      # Main pipeline runner
├── dataset/
│   ├── data_joined.csv          # Raw IMDb data
│   └── data_cleaned.csv         # Cleaned data (generated)
├── models/
│   └── box_office_model.pkl     # Trained model (generated)
├── demo/plots/                  # EDA visualizations (generated)
├── src/
│   ├── pipeline.py              # Data cleaning + model training
│   ├── eda_visualizations.py    # EDA and plots
│   ├── gradio_app.py            # Interactive web interface
│   └── crawler/                 # IMDb data crawler
└── requirements.txt             # Python dependencies
```

## 🎯 Features

### Data Pipeline (`src/pipeline.py`)
- Cleans raw IMDb data (budgets, gross, runtime, ratings)
- Extracts features from cast, crew, genres, keywords
- Trains Random Forest model
- Saves trained model and metrics

### EDA Visualizations (`src/eda_visualizations.py`)
- Gross revenue distributions
- Budget vs Gross analysis
- Genre performance analysis
- Rating analysis
- Correlation heatmaps
- ROI analysis

### Gradio App (`src/gradio_app.py`)
- **Prediction Tab**: Enter movie details, get box office prediction
- **EDA Tab**: Interactive visualizations
- **Metrics Tab**: Model performance and feature importance
- **Dataset Tab**: View sample data

## 📊 Model Details

- **Algorithm**: Random Forest Regressor
- **Target**: Worldwide Gross Revenue
- **Features**: Budget, Runtime, Rating, Cast/Crew counts, Genre, Year, etc.
- **Evaluation**: R², MAE, RMSE metrics

## 🕷️ Crawl More Data (Optional)

To crawl additional movies from IMDb:

```bash
cd src/crawler
bash run_full_crawl.sh 100  # Crawl 100 movies
```

Then re-run the pipeline to retrain with new data.

## 💡 Usage Example

```python
# Run pipeline programmatically
from src.pipeline import BoxOfficePipeline

pipeline = BoxOfficePipeline('dataset/data_joined.csv')
pipeline.run_full_pipeline()
```

## 🎬 Making Predictions

The Gradio app provides an interactive interface, or use the model directly:

```python
import pickle
import numpy as np

# Load model
with open('models/box_office_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Make prediction
features = [100000000, 120, 7.5, 500000, 10, 15, 2, 20, 1, 1, 2024, 0]  # Example
prediction = model_data['model'].predict([features])[0]
print(f"Predicted Gross: ${prediction:,.0f}")
```

## 📝 Notes

- Dataset includes 26+ movies from IMDb
- Model performance depends on data quality and size
- Crawl more data for better predictions
- All monetary values in USD

---

**Created**: 2024
**Framework**: Python + scikit-learn + Gradio
