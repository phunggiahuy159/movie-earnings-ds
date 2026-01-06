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

# Import feature engineering
try:
    from src.feature_engineering import FeatureEngineer
except:
    from feature_engineering import FeatureEngineer


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
    
    def normalize_studio_names(self):
        """Standardize studio names for consistency"""
        print("\nNormalizing studio names...")
        
        if 'Studios' not in self.df_clean.columns:
            return self
        
        # Studio name mappings
        studio_mappings = {
            'Warner Bros.': 'Warner Bros',
            'Warner Brothers': 'Warner Bros',
            'WB': 'Warner Bros',
            'Walt Disney Pictures': 'Walt Disney',
            'Disney': 'Walt Disney',
            '20th Century Fox': '20th Century',
            'Twentieth Century Fox': '20th Century',
            'Columbia Pictures': 'Columbia',
            'Universal Pictures': 'Universal',
            'Paramount Pictures': 'Paramount',
        }
        
        def normalize_studios(studio_str):
            if pd.isna(studio_str) or studio_str == '':
                return studio_str
            studios = [s.strip() for s in str(studio_str).split(',')]
            normalized = []
            for studio in studios:
                # Apply mappings
                normalized_name = studio_mappings.get(studio, studio)
                normalized.append(normalized_name)
            return ','.join(normalized)
        
        self.df_clean['Studios'] = self.df_clean['Studios'].apply(normalize_studios)
        print("✓ Studio names normalized")
        return self
    
    def standardize_genre_names(self):
        """Standardize genre names (case, spelling)"""
        print("\nStandardizing genre names...")
        
        if 'Genre' not in self.df_clean.columns:
            return self
        
        def standardize_genres(genre_str):
            if pd.isna(genre_str) or genre_str == '':
                return genre_str
            genres = [g.strip().title() for g in str(genre_str).split(',')]
            # Remove duplicates while preserving order
            seen = set()
            unique_genres = []
            for g in genres:
                if g not in seen:
                    seen.add(g)
                    unique_genres.append(g)
            return ','.join(unique_genres)
        
        self.df_clean['Genre'] = self.df_clean['Genre'].apply(standardize_genres)
        # Update Primary_Genre as well
        self.df_clean['Primary_Genre'] = self.df_clean['Genre'].fillna('Unknown').apply(
            lambda x: x.split(',')[0].strip() if x else 'Unknown'
        )
        print("✓ Genre names standardized")
        return self
    
    def handle_outliers(self, column, method='iqr', factor=1.5):
        """Cap outliers using IQR or percentile method"""
        if column not in self.df_clean.columns:
            return self
        
        col_data = pd.to_numeric(self.df_clean[column], errors='coerce')
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Cap values
            before_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            col_data = col_data.clip(lower=lower_bound, upper=upper_bound)
            self.df_clean[column] = col_data
            
            if before_count > 0:
                print(f"  - Capped {before_count} outliers in {column}")
        
        elif method == 'percentile':
            lower = col_data.quantile(0.01)
            upper = col_data.quantile(0.99)
            before_count = ((col_data < lower) | (col_data > upper)).sum()
            col_data = col_data.clip(lower=lower, upper=upper)
            self.df_clean[column] = col_data
            
            if before_count > 0:
                print(f"  - Capped {before_count} outliers in {column}")
        
        return self
    
    def impute_missing_by_genre(self, column):
        """Impute missing values using median per genre"""
        if column not in self.df_clean.columns or 'Primary_Genre' not in self.df_clean.columns:
            return self
        
        missing_before = self.df_clean[column].isnull().sum()
        if missing_before == 0:
            return self
        
        # Calculate median per genre
        genre_medians = self.df_clean.groupby('Primary_Genre')[column].median()
        
        # Fill missing values with genre median
        def fill_with_genre_median(row):
            if pd.isna(row[column]):
                genre = row['Primary_Genre']
                if genre in genre_medians:
                    return genre_medians[genre]
                else:
                    return self.df_clean[column].median()  # Overall median as fallback
            return row[column]
        
        self.df_clean[column] = self.df_clean.apply(fill_with_genre_median, axis=1)
        missing_after = self.df_clean[column].isnull().sum()
        
        print(f"  - Imputed {missing_before - missing_after} missing values in {column}")
        return self
    
    def validate_release_dates(self):
        """Validate release dates (filter out future dates)"""
        if 'Release_Data' not in self.df_clean.columns:
            return self
        
        print("\nValidating release dates...")
        release_dates = pd.to_datetime(self.df_clean['Release_Data'], errors='coerce')
        future_dates = (release_dates > pd.Timestamp.now()).sum()
        
        if future_dates > 0:
            print(f"  - Found {future_dates} future release dates, setting to NaT")
            mask = release_dates > pd.Timestamp.now()
            self.df_clean.loc[mask, 'Release_Data'] = pd.NaT
            self.df_clean.loc[mask, 'Release_Year'] = np.nan
        
        print("✓ Release dates validated")
        return self
    
    def validate_value_ranges(self):
        """Validate and fix value ranges"""
        print("\nValidating value ranges...")
        
        # Rating should be 0-10
        if 'Rating' in self.df_clean.columns:
            invalid_ratings = ((self.df_clean['Rating'] < 0) | (self.df_clean['Rating'] > 10)).sum()
            if invalid_ratings > 0:
                print(f"  - Fixing {invalid_ratings} invalid ratings")
                self.df_clean.loc[(self.df_clean['Rating'] < 0) | (self.df_clean['Rating'] > 10), 'Rating'] = np.nan
        
        # Runtime should be 40-300 minutes
        if 'Runtime' in self.df_clean.columns:
            invalid_runtime = ((self.df_clean['Runtime'] < 40) | (self.df_clean['Runtime'] > 300)).sum()
            if invalid_runtime > 0:
                print(f"  - Capping {invalid_runtime} invalid runtimes")
                self.df_clean['Runtime'] = self.df_clean['Runtime'].clip(lower=40, upper=300)
        
        # Budget and Gross should be positive
        for col in ['Budget', 'Gross_worldwide']:
            if col in self.df_clean.columns:
                negative_count = (self.df_clean[col] < 0).sum()
                if negative_count > 0:
                    print(f"  - Setting {negative_count} negative {col} values to NaN")
                    self.df_clean.loc[self.df_clean[col] < 0, col] = np.nan
        
        print("✓ Value ranges validated")
        return self
    
    def advanced_cleaning(self):
        """Execute all advanced cleaning steps"""
        print("\n" + "="*60)
        print("ADVANCED DATA CLEANING")
        print("="*60)
        
        (self
            .normalize_studio_names()
            .standardize_genre_names()
            .validate_value_ranges()
            .validate_release_dates()
            .impute_missing_by_genre('Budget')
            .impute_missing_by_genre('Runtime')
            .handle_outliers('Budget', method='percentile')
            .handle_outliers('Gross_worldwide', method='percentile'))
        
        print("\n✓ Advanced cleaning completed")
        print("="*60)
        return self
    
    def engineer_features(self):
        """Apply advanced feature engineering"""
        print("\nApplying feature engineering...")
        engineer = FeatureEngineer(self.df_clean)
        self.df_clean = engineer.run_full_engineering()
        return self
    
    def prepare_features(self):
        """Prepare features for ML model - using ALL engineered features"""
        print("\nPreparing features for modeling...")
        
        # Drop rows with missing target variable
        df = self.df_clean.dropna(subset=['Gross_worldwide']).copy()
        print(f"Movies with Gross_worldwide: {len(df)}")
        
        # Auto-select numeric and engineered boolean features
        # Exclude non-feature columns
        exclude_cols = [
            'Movie_ID', 'Movie_Title', 'Cast', 'Crew', 'Studios', 'Genre', 
            'Keywords', 'Languages', 'Countries', 'Filming_Location', 
            'Release_Data', 'Gross_worldwide', 'ROI',
            'Release_Data_dt', 'First_Director', 'Lead_Actor', 'Primary_Studio',
            'Budget_Tier', 'Rating_Bucket', 'Primary_Genre', 'Primary_Country'
        ]
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns and target
        feature_cols = [col for col in numeric_cols 
                       if col not in exclude_cols and col != 'Gross_worldwide']
        
        print(f"Selected {len(feature_cols)} features for modeling")
        
        # Keep only rows with required base features
        df_model = df[feature_cols + ['Gross_worldwide']].copy()
        
        # Drop rows with too many missing values (>30% of features missing)
        threshold = len(feature_cols) * 0.7
        df_model = df_model.dropna(thresh=threshold)
        
        # CRITICAL: Fill ALL remaining NaN and inf values
        print(f"  - Filling missing values...")
        for col in feature_cols:
            # Ensure column is numeric
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
            # Replace inf with NaN
            df_model[col] = df_model[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median (or 0 if all NaN)
            median_val = df_model[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            df_model[col] = df_model[col].fillna(fill_val)
        
        # Final verification: drop any rows that still have NaN
        initial_rows = len(df_model)
        df_model = df_model.dropna(subset=feature_cols)
        dropped_rows = initial_rows - len(df_model)
        if dropped_rows > 0:
            print(f"  - Dropped {dropped_rows} rows with remaining NaN values")
        
        print(f"Complete data for modeling: {len(df_model)} movies")
        print(f"Features being used: {len(feature_cols)}")
        
        # Verify no NaN values remain
        nan_count = df_model[feature_cols].isnull().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️ Warning: {nan_count} NaN values still present, dropping affected rows...")
            df_model = df_model.dropna(subset=feature_cols)
            print(f"  ✓ Final data for modeling: {len(df_model)} movies")
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Prepare X and y
        X = df_model[self.feature_names]
        y = df_model['Gross_worldwide']
        
        return X, y
    
    def train_model(self, X, y):
        """Train and compare multiple models"""
        print("\n" + "="*70)
        print(" MODEL TRAINING & COMPARISON")
        print("="*70)
        
        # Import model training module
        try:
            from src.model_training import ModelTrainer
        except:
            from model_training import ModelTrainer
        
        # Train and compare 7 models
        trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
        
        # Define models, cross-validate, and train with tuning
        trainer.define_models()
        trainer.cross_validate_all(cv=3)  # 3-fold for speed with large dataset
        trainer.train_all_models(tune=True)
        trainer.generate_comparison_report()
        trainer.get_feature_importance(top_n=20)
        trainer.save_best_model()
        
        # Store results for compatibility
        self.model = trainer.best_model
        self.model_trainer = trainer
        self.metrics = trainer.results.get(trainer.best_model_name, {})
        
        print("\n✓ Model training completed")
        return self
    
    def save_model(self, model_path='models/box_office_model.pkl'):
        """Save trained model with metadata"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': getattr(self, 'model_trainer', None) and self.model_trainer.best_model_name or 'Random_Forest',
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Best model saved to {model_path}")
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
        self.advanced_cleaning()  # Advanced cleaning step
        self.engineer_features()  # NEW: Feature engineering step
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
