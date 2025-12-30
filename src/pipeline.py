"""
Complete ML Pipeline for Box Office Revenue Prediction
Data Cleaning -> Feature Engineering -> Model Training
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BoxOfficePipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.metrics = {}
        
    def load_data(self):
        """Load raw data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} movies")
        return self
    
    def clean_data(self):
        """Clean and preprocess data"""
        print("\nCleaning data...")
        df = self.df.copy()
        
        # Parse Budget - extract numbers from strings like "$63,000,000"
        def parse_money(value):
            if pd.isna(value) or value == '':
                return np.nan
            if isinstance(value, (int, float)):
                return float(value)
            # Extract numbers and remove commas
            match = re.search(r'\$?([\d,]+)', str(value))
            if match:
                return float(match.group(1).replace(',', ''))
            return np.nan
        
        df['Budget'] = df['Budget'].apply(parse_money)
        df['Gross_worldwide'] = df['Gross_worldwide'].apply(parse_money)
        
        # Parse Runtime - extract minutes
        def parse_runtime(value):
            if pd.isna(value):
                return np.nan
            match = re.search(r'(\d+)', str(value))
            if match:
                return int(match.group(1))
            return np.nan
        
        df['Runtime'] = df['Runtime'].apply(parse_runtime)
        
        # Convert Rating to numeric
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Rating_Count'] = pd.to_numeric(df['Rating_Count'], errors='coerce')
        
        # Extract year from Release_Data
        df['Release_Year'] = pd.to_datetime(df['Release_Data'], errors='coerce').dt.year
        
        # Count features from comma-separated lists
        df['Cast_Count'] = df['Cast'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        df['Crew_Count'] = df['Crew'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        df['Genre_Count'] = df['Genre'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        df['Keywords_Count'] = df['Keywords'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        df['Languages_Count'] = df['Languages'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        df['Countries_Count'] = df['Countries'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        
        # Extract primary genre
        df['Primary_Genre'] = df['Genre'].fillna('Unknown').apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
        
        # ROI calculation (if both Budget and Gross available)
        df['ROI'] = ((df['Gross_worldwide'] - df['Budget']) / df['Budget'] * 100).replace([np.inf, -np.inf], np.nan)
        
        print(f"Cleaned data: {len(df)} movies")
        print(f"Missing values:\n{df[['Budget', 'Gross_worldwide', 'Runtime', 'Rating']].isnull().sum()}")
        
        self.df_clean = df
        return self
    
    def prepare_features(self):
        """Prepare features for ML model"""
        print("\nPreparing features...")
        
        # Drop rows with missing target variable
        df = self.df_clean.dropna(subset=['Gross_worldwide']).copy()
        print(f"Movies with Gross_worldwide: {len(df)}")
        
        # Select features for modeling
        feature_cols = [
            'Budget', 'Runtime', 'Rating', 'Rating_Count',
            'Cast_Count', 'Crew_Count', 'Genre_Count', 'Keywords_Count',
            'Languages_Count', 'Countries_Count', 'Release_Year', 'Primary_Genre'
        ]
        
        # Keep only rows with all required features
        df_model = df[feature_cols + ['Gross_worldwide']].copy()
        df_model = df_model.dropna(subset=['Budget', 'Runtime', 'Rating', 'Rating_Count'])
        
        print(f"Complete data for modeling: {len(df_model)} movies")
        
        # Encode categorical features
        if 'Primary_Genre' in df_model.columns:
            le = LabelEncoder()
            df_model['Primary_Genre_Encoded'] = le.fit_transform(df_model['Primary_Genre'])
            self.label_encoders['Primary_Genre'] = le
        
        # Prepare X and y
        self.feature_names = [
            'Budget', 'Runtime', 'Rating', 'Rating_Count',
            'Cast_Count', 'Crew_Count', 'Genre_Count', 'Keywords_Count',
            'Languages_Count', 'Countries_Count', 'Release_Year', 'Primary_Genre_Encoded'
        ]
        
        X = df_model[self.feature_names]
        y = df_model['Gross_worldwide']
        
        return X, y
    
    def train_model(self, X, y):
        """Train Random Forest model"""
        print("\nTraining model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} movies")
        print(f"Test set: {len(X_test)} movies")
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Model Performance ===")
        print(f"Train R²: {self.metrics['train']['r2']:.4f}")
        print(f"Test R²: {self.metrics['test']['r2']:.4f}")
        print(f"Test MAE: ${self.metrics['test']['mae']:,.0f}")
        print(f"Test RMSE: ${self.metrics['test']['rmse']:,.0f}")
        
        print("\n=== Top 5 Feature Importance ===")
        print(feature_importance.head())
        
        return self
    
    def save_model(self, model_path='models/box_office_model.pkl'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {model_path}")
        return self
    
    def save_clean_data(self, output_path='dataset/data_cleaned.csv'):
        """Save cleaned dataset"""
        self.df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return self
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        print("="*60)
        print("BOX OFFICE PREDICTION PIPELINE")
        print("="*60)
        
        self.load_data()
        self.clean_data()
        self.save_clean_data()
        
        X, y = self.prepare_features()
        self.train_model(X, y)
        self.save_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self


if __name__ == "__main__":
    # Run pipeline
    pipeline = BoxOfficePipeline('dataset/data_joined.csv')
    pipeline.run_full_pipeline()
