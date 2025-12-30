"""
Main script to run the complete ML pipeline
Usage: python main.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import BoxOfficePipeline
from src.eda_visualizations import EDAVisualizer


def main():
    """Run complete workflow"""
    
    print("\n" + "="*70)
    print(" BOX OFFICE PREDICTION - COMPLETE ML PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Pipeline (Cleaning + Feature Engineering + Model Training)
    print("STEP 1: Running Data Pipeline...")
    print("-" * 70)
    pipeline = BoxOfficePipeline('dataset/data_joined.csv')
    pipeline.run_full_pipeline()
    
    # Step 2: Generate EDA Visualizations
    print("\n" + "="*70)
    print("STEP 2: Generating Visualizations...")
    print("-" * 70)
    visualizer = EDAVisualizer('dataset/data_cleaned.csv')
    visualizer.generate_all_plots()
    
    # Step 3: Instructions for Gradio App
    print("\n" + "="*70)
    print("STEP 3: Launch Gradio App")
    print("-" * 70)
    print("\nTo start the interactive Gradio app, run:")
    print("  python src/gradio_app.py")
    print("\nOr run:")
    print("  python -m src.gradio_app")
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - dataset/data_cleaned.csv (cleaned dataset)")
    print("  - models/box_office_model.pkl (trained model)")
    print("  - demo/plots/*.png (EDA visualizations)")
    print("\nNext: Launch the Gradio app to interact with the model")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
