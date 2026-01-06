"""
Model Evaluation and Validation
Comprehensive analysis of model performance with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model, X_train, X_test, y_train, y_test, model_name='Model'):
        """
        Initialize evaluator
        
        Args:
            model: Trained sklearn model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            model_name: Name of the model for display
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        
        # Generate predictions
        self.y_train_pred = model.predict(X_train)
        self.y_test_pred = model.predict(X_test)
        
        print(f"ModelEvaluator initialized for {model_name}")
    
    def calculate_all_metrics(self):
        """Compute comprehensive metrics for both sets"""
        print("\n=== PERFORMANCE METRICS ===")
        
        metrics = {}
        
        for set_name, y_true, y_pred in [
            ('Train', self.y_train, self.y_train_pred),
            ('Test', self.y_test, self.y_test_pred)
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (handle zeros)
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            # Max error
            max_error = np.max(np.abs(y_true - y_pred))
            
            metrics[set_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Max_Error': max_error
            }
            
            print(f"\n{set_name} Set:")
            print(f"  R² Score:    {r2:.4f}")
            print(f"  MAE:         ${mae:,.0f}")
            print(f"  RMSE:        ${rmse:,.0f}")
            print(f"  MAPE:        {mape:.2f}%")
            print(f"  Max Error:   ${max_error:,.0f}")
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv('demo/model_metrics.csv')
        print("\n✓ Metrics saved to demo/model_metrics.csv")
        
        return metrics
    
    def plot_residuals(self, save_path='demo/plots/evaluation/'):
        """Residual plot and histogram"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        residuals = self.y_test - self.y_test_pred
        
        axes[0].scatter(self.y_test_pred / 1e6, residuals / 1e6, alpha=0.5, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Gross (Millions $)')
        axes[0].set_ylabel('Residuals (Millions $)')
        axes[0].set_title(f'Residual Plot - {self.model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals / 1e6, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals (Millions $)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}residuals.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Residual plots saved to {save_path}residuals.png")
        return fig
    
    def plot_prediction_vs_actual(self, save_path='demo/plots/evaluation/'):
        """Scatter plot with perfect prediction line"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter plot
        ax.scatter(self.y_test / 1e6, self.y_test_pred / 1e6, 
                  alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_test_pred.min()) / 1e6
        max_val = max(self.y_test.max(), self.y_test_pred.max()) / 1e6
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(self.y_test, self.y_test_pred)
        
        ax.set_xlabel('Actual Gross (Millions $)')
        ax.set_ylabel('Predicted Gross (Millions $)')
        ax.set_title(f'Predicted vs Actual - {self.model_name}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}pred_vs_actual.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Prediction plot saved to {save_path}pred_vs_actual.png")
        return fig
    
    def plot_learning_curve(self, save_path='demo/plots/evaluation/'):
        """Learning curve (training size vs score)"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\nGenerating learning curve (this may take a moment)...")
        
        # Combine train and test for learning curve
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=3, 
            scoring='r2',
            n_jobs=-1
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('R² Score')
        ax.set_title(f'Learning Curve - {self.model_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}learning_curve.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Learning curve saved to {save_path}learning_curve.png")
        return fig
    
    def plot_validation_curve(self, param_name, param_range, save_path='demo/plots/evaluation/'):
        """Validation curve for hyperparameter"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nGenerating validation curve for {param_name}...")
        
        # Combine train and test
        X = pd.concat([self.X_train, self.X_test])
        y = pd.concat([self.y_train, self.y_test])
        
        try:
            train_scores, val_scores = validation_curve(
                self.model, X, y,
                param_name=param_name,
                param_range=param_range,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                            alpha=0.1, color='blue')
            
            ax.plot(param_range, val_mean, 'o-', color='red', label='Validation score')
            ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                            alpha=0.1, color='red')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('R² Score')
            ax.set_title(f'Validation Curve - {param_name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}validation_curve_{param_name}.png', dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Validation curve saved to {save_path}validation_curve_{param_name}.png")
            return fig
            
        except Exception as e:
            print(f"⚠ Could not generate validation curve: {e}")
            return None
    
    def analyze_errors_by_category(self, df, category_col='Primary_Genre', save_path='demo/plots/evaluation/'):
        """Error analysis grouped by category"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if category_col not in df.columns:
            print(f"⚠ Column {category_col} not found")
            return None
        
        # Get test indices
        test_indices = self.X_test.index if hasattr(self.X_test, 'index') else range(len(self.X_test))
        
        # Create error DataFrame
        error_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': self.y_test_pred,
            'Error': np.abs(self.y_test.values - self.y_test_pred),
            'Percent_Error': np.abs((self.y_test.values - self.y_test_pred) / self.y_test.values * 100)
        }, index=test_indices)
        
        # Add category
        error_df[category_col] = df.loc[test_indices, category_col].values
        
        # Group by category
        error_summary = error_df.groupby(category_col).agg({
            'Error': ['mean', 'median', 'std', 'count'],
            'Percent_Error': 'mean'
        }).round(2)
        
        error_summary.columns = ['MAE', 'Median_AE', 'Std_Error', 'Count', 'MAPE']
        error_summary = error_summary.sort_values('MAE', ascending=False).head(10)
        
        print(f"\n=== ERROR ANALYSIS BY {category_col} (Top 10) ===")
        print(error_summary)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(error_summary))
        ax.bar(x, error_summary['MAE'] / 1e6, alpha=0.7)
        ax.set_xlabel(category_col)
        ax.set_ylabel('Mean Absolute Error (Millions $)')
        ax.set_title(f'Prediction Error by {category_col}')
        ax.set_xticks(x)
        ax.set_xticklabels(error_summary.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}error_by_{category_col}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error analysis saved to {save_path}error_by_{category_col}.png")
        return error_summary
    
    def generate_evaluation_report(self, df=None, save_path='demo/plots/evaluation/'):
        """Full evaluation report with all metrics and plots"""
        print("\n" + "="*70)
        print(f" COMPREHENSIVE EVALUATION: {self.model_name}")
        print("="*70)
        
        # Calculate metrics
        metrics = self.calculate_all_metrics()
        
        # Generate plots
        print("\nGenerating evaluation plots...")
        self.plot_residuals(save_path)
        self.plot_prediction_vs_actual(save_path)
        
        try:
            self.plot_learning_curve(save_path)
        except Exception as e:
            print(f"⚠ Learning curve failed: {e}")
        
        # Error by category (if DataFrame provided)
        if df is not None:
            for cat_col in ['Primary_Genre', 'Budget_Tier']:
                if cat_col in df.columns:
                    try:
                        self.analyze_errors_by_category(df, cat_col, save_path)
                    except Exception as e:
                        print(f"⚠ Error analysis for {cat_col} failed: {e}")
        
        print("\n" + "="*70)
        print(" EVALUATION COMPLETED")
        print("="*70)
        
        return metrics


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("This module provides comprehensive model evaluation")
    print("\nUsage:")
    print("  from model_evaluation import ModelEvaluator")
    print("  evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)")
    print("  evaluator.generate_evaluation_report()")

