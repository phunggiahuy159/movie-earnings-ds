"""
Model Training and Selection for Box Office Prediction
Compares multiple algorithms and selects the best performer
"""

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and compare multiple regression models"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize trainer with features and target
        
        Args:
            X: Feature matrix (DataFrame or array)
            y: Target vector (Series or array)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        print(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test")
    
    def define_models(self):
        """Define all models with hyperparameter grids for tuning"""
        print("\n=== DEFINING MODELS ===")
        
        self.models = {
            'Linear_Regression': {
                'model': LinearRegression(),
                'params': {}  # No hyperparameters to tune
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.2, 0.5, 0.8]
                }
            },
            'Decision_Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Random_Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5]
                }
            }
        }
        
        print(f"✓ Defined {len(self.models)} models for comparison")
        return self
    
    def cross_validate_all(self, cv=5):
        """Perform cross-validation for all models"""
        print("\n=== CROSS-VALIDATION (Initial Models) ===")
        
        cv_results = []
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            
            print(f"\nTesting {name}...")
            try:
                # Use default parameters first
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                        cv=cv, scoring='r2', n_jobs=-1)
                
                cv_results.append({
                    'Model': name,
                    'CV_R2_Mean': scores.mean(),
                    'CV_R2_Std': scores.std(),
                    'CV_R2_Min': scores.min(),
                    'CV_R2_Max': scores.max()
                })
                
                print(f"  R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                cv_results.append({
                    'Model': name,
                    'CV_R2_Mean': np.nan,
                    'CV_R2_Std': np.nan,
                    'CV_R2_Min': np.nan,
                    'CV_R2_Max': np.nan
                })
        
        self.results['cv_results'] = pd.DataFrame(cv_results).sort_values('CV_R2_Mean', ascending=False)
        print("\n" + self.results['cv_results'].to_string(index=False))
        
        return self
    
    def tune_hyperparameters(self, model_name, cv=5):
        """Perform GridSearchCV for specific model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model_dict = self.models[model_name]
        model = model_dict['model']
        params = model_dict['params']
        
        if not params:
            print(f"{model_name} has no hyperparameters to tune")
            return model
        
        print(f"\nTuning {model_name}...")
        print(f"  Testing {np.prod([len(v) for v in params.values()])} combinations...")
        
        try:
            grid_search = GridSearchCV(
                model,
                params,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"  ✓ Best params: {grid_search.best_params_}")
            print(f"  ✓ Best MAE: ${-grid_search.best_score_:,.0f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"  ✗ Tuning failed: {e}")
            return model
    
    def train_all_models(self, tune=True):
        """Train all models (with optional hyperparameter tuning)"""
        print("\n" + "="*70)
        print(" TRAINING ALL MODELS")
        print("="*70)
        
        trained_models = {}
        
        for name, model_dict in self.models.items():
            print(f"\n[{name}]")
            
            try:
                # Tune hyperparameters if requested
                if tune and model_dict['params']:
                    model = self.tune_hyperparameters(name, cv=3)  # 3-fold for speed
                else:
                    model = model_dict['model']
                
                # Train on full training set
                print(f"  Training on {len(self.X_train)} samples...")
                model.fit(self.X_train, self.y_train)
                
                # Evaluate
                train_metrics = self.evaluate_model(model, self.X_train, self.y_train)
                test_metrics = self.evaluate_model(model, self.X_test, self.y_test)
                
                print(f"  Train R²: {train_metrics['R2']:.4f}, MAE: ${train_metrics['MAE']:,.0f}")
                print(f"  Test R²:  {test_metrics['R2']:.4f}, MAE: ${test_metrics['MAE']:,.0f}")
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train': train_metrics,
                    'test': test_metrics
                }
                
                trained_models[name] = model
                
            except Exception as e:
                print(f"  ✗ Training failed: {e}")
        
        # Find best model
        best_r2 = -np.inf
        for name, results in self.results.items():
            if isinstance(results, dict) and 'test' in results:
                if results['test']['R2'] > best_r2:
                    best_r2 = results['test']['R2']
                    self.best_model_name = name
                    self.best_model = results['model']
        
        if self.best_model_name:
            print(f"\n{'='*70}")
            print(f" BEST MODEL: {self.best_model_name}")
            print(f" Test R² = {best_r2:.4f}")
            print(f"{'='*70}")
        
        return self
    
    def evaluate_model(self, model, X, y):
        """Calculate all evaluation metrics"""
        try:
            preds = model.predict(X)
            
            # Handle potential division by zero in MAPE
            mask = y != 0
            mape = np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100 if mask.sum() > 0 else np.nan
            
            return {
                'MAE': mean_absolute_error(y, preds),
                'RMSE': np.sqrt(mean_squared_error(y, preds)),
                'R2': r2_score(y, preds),
                'MAPE': mape
            }
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'R2': np.nan,
                'MAPE': np.nan
            }
    
    def generate_comparison_report(self):
        """Create model comparison DataFrame"""
        print("\n=== MODEL COMPARISON REPORT ===")
        
        comparison = []
        
        for name, results in self.results.items():
            if isinstance(results, dict) and 'test' in results:
                comparison.append({
                    'Model': name,
                    'Train_R2': results['train']['R2'],
                    'Test_R2': results['test']['R2'],
                    'Test_MAE': results['test']['MAE'],
                    'Test_RMSE': results['test']['RMSE'],
                    'Test_MAPE': results['test']['MAPE'],
                    'Overfit': results['train']['R2'] - results['test']['R2']
                })
        
        comparison_df = pd.DataFrame(comparison).sort_values('Test_R2', ascending=False)
        
        # Save to CSV
        comparison_df.to_csv('demo/model_comparison.csv', index=False)
        print(comparison_df.to_string(index=False))
        print("\n✓ Saved to demo/model_comparison.csv")
        
        self.results['comparison'] = comparison_df
        return comparison_df
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from best model (if available)"""
        if not self.best_model or not self.feature_names:
            return None
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(top_n)
                
                print(f"\n=== TOP {top_n} FEATURE IMPORTANCE ({self.best_model_name}) ===")
                print(importance_df.to_string(index=False))
                
                return importance_df
            
            elif hasattr(self.best_model, 'coef_'):
                coefs = np.abs(self.best_model.coef_)
                
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Abs_Coefficient': coefs
                }).sort_values('Abs_Coefficient', ascending=False).head(top_n)
                
                print(f"\n=== TOP {top_n} FEATURE COEFFICIENTS ({self.best_model_name}) ===")
                print(importance_df.to_string(index=False))
                
                return importance_df
        
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
        
        return None
    
    def save_best_model(self, path='models/box_office_model.pkl'):
        """Save best model with metadata"""
        if not self.best_model:
            print("No model trained yet")
            return
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'metrics': self.results.get(self.best_model_name, {})
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Best model ({self.best_model_name}) saved to {path}")
        return self


def run_model_comparison(X, y):
    """Convenient function to run full model comparison"""
    print("="*70)
    print(" MODEL TRAINING & COMPARISON PIPELINE")
    print("="*70)
    
    trainer = ModelTrainer(X, y)
    
    (trainer
        .define_models()
        .cross_validate_all()
        .train_all_models(tune=True)
        .generate_comparison_report()
        .get_feature_importance()
        .save_best_model())
    
    print("\n" + "="*70)
    print(" MODEL COMPARISON COMPLETED")
    print("="*70)
    
    return trainer


if __name__ == "__main__":
    print("Model Training Module")
    print("This module trains and compares 7 regression models")
    print("\nUsage:")
    print("  from model_training import ModelTrainer, run_model_comparison")
    print("  trainer = run_model_comparison(X, y)")

