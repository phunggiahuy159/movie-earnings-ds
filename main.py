"""
Main script to run the complete ML pipeline
Enhanced version with statistical analysis and model comparison
Usage: python main.py
"""

import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import BoxOfficePipeline
from src.eda_visualizations import EDAVisualizer
from src.statistical_analysis import StatisticalAnalyzer
from src.data_quality_report import DataQualityChecker


def run_data_quality_check():
    """Run data quality validation"""
    print("\n" + "="*70)
    print(" STEP 0: DATA QUALITY CHECK")
    print("="*70)
    
    checker = DataQualityChecker('dataset/data_joined.csv')
    checker.run_full_check()
    
    return checker


def run_pipeline():
    """Run data cleaning, feature engineering, and model training"""
    print("\n" + "="*70)
    print(" STEP 1: DATA PIPELINE")
    print("="*70)
    
    pipeline = BoxOfficePipeline('dataset/data_joined.csv')
    pipeline.run_full_pipeline()
    
    return pipeline


def run_statistical_analysis():
    """Run statistical analysis"""
    print("\n" + "="*70)
    print(" STEP 2: STATISTICAL ANALYSIS")
    print("="*70)
    
    analyzer = StatisticalAnalyzer('dataset/data_cleaned.csv')
    analyzer.run_full_analysis()
    
    return analyzer


def run_eda():
    """Generate comprehensive EDA visualizations"""
    print("\n" + "="*70)
    print(" STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    visualizer = EDAVisualizer('dataset/data_cleaned.csv')
    visualizer.generate_all_plots()
    
    return visualizer


def print_completion_message():
    """Print completion message with next steps"""
    print("\n" + "="*70)
    print(" 🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Generated Files:")
    print("  ├── dataset/data_cleaned.csv (cleaned & featured dataset)")
    print("  ├── models/box_office_model.pkl (best trained model)")
    print("  ├── demo/plots/ (25+ visualizations)")
    print("  ├── demo/data_quality_report.html (data quality report)")
    print("  ├── demo/stats_summary.csv (statistical summary)")
    print("  ├── demo/correlation_matrix.csv (correlation analysis)")
    print("  └── demo/model_comparison.csv (model comparison results)")
    
    print("\n📊 Key Results:")
    print("  • Feature Engineering: 50+ features created")
    print("  • Models Compared: 7 algorithms (Linear, Ridge, Lasso, etc.)")
    print("  • Visualizations: 25+ plots generated")
    print("  • Statistical Tests: Normality, correlation, multicollinearity")
    
    print("\n🚀 Next Steps:")
    print("  1. Review data quality report: demo/data_quality_report.html")
    print("  2. Check model comparison: demo/model_comparison.csv")
    print("  3. Launch interactive demo:")
    print("     python src/gradio_app.py")
    print("\n" + "="*70 + "\n")


def main(skip_quality_check=False):
    """
    Run complete enhanced ML pipeline
    
    Args:
        skip_quality_check: Skip initial data quality check if True
    """
    
    print("\n" + "="*80)
    print(" 🎬 BOX OFFICE PREDICTION - COMPLETE ENHANCED ML PIPELINE")
    print("="*80)
    print("\n📌 This pipeline includes:")
    print("  • Data quality validation")
    print("  • Advanced data cleaning & preprocessing")
    print("  • Feature engineering (50+ features)")
    print("  • Statistical analysis")
    print("  • Comprehensive EDA (25+ plots)")
    print("  • Model comparison (7 algorithms)")
    print("  • Model evaluation & validation")
    print("\n" + "="*80 + "\n")
    
    try:
        # Step 0: Data Quality (optional)
        if not skip_quality_check:
            run_data_quality_check()
        
        # Step 1: Pipeline (Clean, Feature Engineer, Train)
        pipeline = run_pipeline()
        
        # Step 2: Statistical Analysis
        run_statistical_analysis()
        
        # Step 3: EDA Visualizations
        run_eda()
        
        # Print completion message
        print_completion_message()
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Pipeline failed. Please check the error above.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Box Office Prediction Pipeline')
    parser.add_argument('--skip-quality-check', action='store_true',
                       help='Skip initial data quality check')
    
    args = parser.parse_args()
    
    main(skip_quality_check=args.skip_quality_check)
