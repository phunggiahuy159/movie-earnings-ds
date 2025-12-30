"""
Gradio Interactive App for Box Office Prediction
Includes EDA visualizations, model predictions, and performance metrics
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
import sys
sys.path.append('src')
from eda_visualizations import EDAVisualizer


class BoxOfficeApp:
    def __init__(self):
        self.model_data = None
        self.df = None
        self.visualizer = None
        self.load_resources()
        
    def load_resources(self):
        """Load model and data"""
        # Load model
        model_path = 'models/box_office_model.pkl'
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print("Model loaded successfully")
        else:
            print(f"Warning: Model not found at {model_path}")
            
        # Load data
        data_path = 'dataset/data_cleaned.csv'
        if Path(data_path).exists():
            self.df = pd.read_csv(data_path)
            print(f"Data loaded: {len(self.df)} movies")
        else:
            print(f"Warning: Data not found at {data_path}")
            
        # Initialize visualizer
        self.visualizer = EDAVisualizer(data_path)
        if self.df is not None:
            self.visualizer.df = self.df
    
    def predict_gross(self, budget, runtime, rating, rating_count, 
                      cast_count, crew_count, genre_count, keywords_count,
                      languages_count, countries_count, release_year, primary_genre):
        """Make prediction for box office gross"""
        
        if self.model_data is None:
            return "Error: Model not loaded", None
        
        try:
            # Encode genre
            genre_encoder = self.model_data['label_encoders']['Primary_Genre']
            
            if primary_genre not in genre_encoder.classes_:
                primary_genre_encoded = 0  # Default to first class
            else:
                primary_genre_encoded = genre_encoder.transform([primary_genre])[0]
            
            # Create feature vector
            features = np.array([[
                budget, runtime, rating, rating_count,
                cast_count, crew_count, genre_count, keywords_count,
                languages_count, countries_count, release_year, primary_genre_encoded
            ]])
            
            # Make prediction
            model = self.model_data['model']
            prediction = model.predict(features)[0]
            
            # Calculate potential ROI
            roi = ((prediction - budget) / budget * 100) if budget > 0 else 0
            
            result = f"""
### Prediction Results

**Predicted Worldwide Gross:** ${prediction:,.0f}

**Budget:** ${budget:,.0f}

**Expected Profit:** ${(prediction - budget):,.0f}

**ROI:** {roi:.1f}%

---

*Note: This prediction is based on a Random Forest model trained on {len(self.df)} movies.*
"""
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(8, 5))
            
            categories = ['Budget', 'Predicted\nGross']
            values = [budget, prediction]
            colors = ['lightcoral', 'lightgreen']
            
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Amount ($)')
            ax.set_title('Budget vs Predicted Gross')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height/1e6:.1f}M',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            return result, fig
            
        except Exception as e:
            return f"Error making prediction: {str(e)}", None
    
    def show_model_metrics(self):
        """Display model performance metrics"""
        if self.model_data is None:
            return "Model not loaded"
        
        metrics = self.model_data['metrics']
        
        result = f"""
## Model Performance Metrics

### Training Set
- **R² Score:** {metrics['train']['r2']:.4f}
- **MAE:** ${metrics['train']['mae']:,.0f}
- **RMSE:** ${metrics['train']['rmse']:,.0f}

### Test Set
- **R² Score:** {metrics['test']['r2']:.4f}
- **MAE:** ${metrics['test']['mae']:,.0f}
- **RMSE:** ${metrics['test']['rmse']:,.0f}

---

### Feature Importance (Top 5)
"""
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.model_data['feature_names'],
            'Importance': self.model_data['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, row in feature_importance.head(5).iterrows():
            result += f"\n{idx+1}. **{row['Feature']}**: {row['Importance']:.4f}"
        
        return result
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Get available genres
        if self.df is not None:
            genres = sorted(self.df['Primary_Genre'].unique().tolist())
        else:
            genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Adventure']
        
        with gr.Blocks(title="Box Office Prediction", theme=gr.themes.Soft()) as app:
            
            gr.Markdown("# 🎬 Box Office Revenue Prediction System")
            gr.Markdown("Predict movie box office performance using machine learning")
            
            with gr.Tabs():
                
                # Tab 1: Predictions
                with gr.Tab("🎯 Make Prediction"):
                    gr.Markdown("### Enter Movie Details")
                    
                    with gr.Row():
                        with gr.Column():
                            budget = gr.Number(label="Budget ($)", value=100000000)
                            runtime = gr.Number(label="Runtime (minutes)", value=120)
                            rating = gr.Slider(0, 10, value=7.5, label="IMDb Rating")
                            rating_count = gr.Number(label="Number of Ratings", value=500000)
                            release_year = gr.Number(label="Release Year", value=2024)
                            primary_genre = gr.Dropdown(choices=genres, label="Primary Genre", value=genres[0])
                        
                        with gr.Column():
                            cast_count = gr.Slider(1, 50, value=10, step=1, label="Number of Cast Members")
                            crew_count = gr.Slider(1, 50, value=15, step=1, label="Number of Crew Members")
                            genre_count = gr.Slider(1, 5, value=2, step=1, label="Number of Genres")
                            keywords_count = gr.Slider(0, 100, value=20, step=1, label="Number of Keywords")
                            languages_count = gr.Slider(1, 10, value=1, step=1, label="Number of Languages")
                            countries_count = gr.Slider(1, 10, value=1, step=1, label="Number of Countries")
                    
                    predict_btn = gr.Button("Predict Box Office Gross", variant="primary", size="lg")
                    
                    with gr.Row():
                        prediction_output = gr.Markdown()
                        prediction_plot = gr.Plot()
                    
                    predict_btn.click(
                        fn=self.predict_gross,
                        inputs=[budget, runtime, rating, rating_count, cast_count, crew_count,
                               genre_count, keywords_count, languages_count, countries_count,
                               release_year, primary_genre],
                        outputs=[prediction_output, prediction_plot]
                    )
                
                # Tab 2: EDA Visualizations
                with gr.Tab("📊 Exploratory Analysis"):
                    gr.Markdown("### Data Visualizations")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Gross Distribution")
                            if self.visualizer.df is not None:
                                fig1 = self.visualizer.plot_gross_distribution()
                                gr.Plot(value=fig1)
                                plt.close(fig1)
                        
                        with gr.Column():
                            gr.Markdown("#### Budget vs Gross")
                            if self.visualizer.df is not None:
                                fig2 = self.visualizer.plot_budget_vs_gross()
                                gr.Plot(value=fig2)
                                plt.close(fig2)
                    
                    with gr.Row():
                        gr.Markdown("#### Genre Analysis")
                        if self.visualizer.df is not None:
                            fig3 = self.visualizer.plot_genre_analysis()
                            gr.Plot(value=fig3)
                            plt.close(fig3)
                    
                    with gr.Row():
                        gr.Markdown("#### Rating Analysis")
                        if self.visualizer.df is not None:
                            fig4 = self.visualizer.plot_rating_analysis()
                            gr.Plot(value=fig4)
                            plt.close(fig4)
                
                # Tab 3: Model Metrics
                with gr.Tab("📈 Model Performance"):
                    gr.Markdown("### Model Evaluation Metrics")
                    metrics_output = gr.Markdown(value=self.show_model_metrics())
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Correlation Heatmap")
                    if self.visualizer.df is not None:
                        fig5 = self.visualizer.plot_correlation_heatmap()
                        gr.Plot(value=fig5)
                        plt.close(fig5)
                
                # Tab 4: Dataset Info
                with gr.Tab("ℹ️ Dataset Info"):
                    gr.Markdown("### Dataset Overview")
                    
                    if self.df is not None:
                        info_text = f"""
**Total Movies:** {len(self.df)}

**Features:** Budget, Runtime, Rating, Rating Count, Cast Count, Crew Count, 
Genre Count, Keywords Count, Languages Count, Countries Count, Release Year, Primary Genre

**Target Variable:** Gross Worldwide Revenue

#### Sample Statistics
"""
                        gr.Markdown(info_text)
                        
                        # Show sample data
                        sample_cols = ['Movie_Title', 'Budget', 'Gross_worldwide', 'Rating', 'Primary_Genre']
                        if all(col in self.df.columns for col in sample_cols):
                            sample_df = self.df[sample_cols].head(10)
                            gr.Dataframe(value=sample_df)
                    else:
                        gr.Markdown("Dataset not loaded")
        
        return app


def main():
    """Run the application"""
    print("="*60)
    print("STARTING BOX OFFICE PREDICTION APP")
    print("="*60)
    
    app = BoxOfficeApp()
    interface = app.create_interface()
    
    print("\nLaunching Gradio interface...")
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
