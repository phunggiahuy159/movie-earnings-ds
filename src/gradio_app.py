"""
Comprehensive Gradio Demo for Box Office Prediction
Showcases: EDA, Statistics, 7 Models, Feature Engineering
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Helper functions
def load_pickle(path):
    """Load pickle file safely"""
    return pickle.load(open(path, 'rb')) if Path(path).exists() else None

def load_csv(path):
    """Load CSV safely"""
    return pd.read_csv(path) if Path(path).exists() else None

def load_html(path):
    """Load HTML file"""
    return open(path, 'r', encoding='utf-8').read() if Path(path).exists() else "<p>File not found</p>"

def load_image(path):
    """Load image safely"""
    return Image.open(path) if Path(path).exists() else None

class BoxOfficeDemo:
    def __init__(self):
        print("Loading resources...")
        self.model = load_pickle('models/box_office_model.pkl')
        self.df = load_csv('dataset/data_cleaned.csv')
        self.comparison = load_csv('demo/model_comparison.csv')
        self.stats = load_csv('demo/stats_summary.csv')
        self.vif = load_csv('demo/vif_report.csv')
        print(f"✓ Loaded: {len(self.df) if self.df is not None else 0} movies, {len(self.model['feature_names']) if self.model else 0} features")
    
    # Tab 1: Overview
    def overview_tab(self):
        """Project overview with key metrics"""
        metrics = f"""
# 🎬 Movie Box Office Prediction
**Advanced ML System with Comprehensive Feature Engineering**

## 🎯 Project Goals
Predict worldwide box office gross using 19,000 movies dataset, demonstrating:
- Data Science Process (Collection → Cleaning → EDA → Modeling → Evaluation)
- Advanced Feature Engineering (52 new features created)
- Model Comparison (7 ML algorithms)
- Statistical Analysis & Visualization

## 📊 Key Results
- **Dataset:** {len(self.df):,} movies ({self.df['Release_Year'].min():.0f}-{self.df['Release_Year'].max():.0f})
- **Features:** {len(self.model['feature_names'])} engineered features
- **Best Model:** {self.model.get('model_name', 'Random_Forest')}
- **Performance:** R² = {self.comparison.iloc[0]['Test_R2']:.4f}, MAE = ${self.comparison.iloc[0]['Test_MAE']:,.0f}
- **Models Compared:** 7 algorithms (Linear, Ridge, Lasso, ElasticNet, Tree, Forest, Boosting)

## 📚 Course Topics Demonstrated
✅ Data Collection & Cleaning  
✅ EDA & Visualization (25+ plots)  
✅ Statistical Analysis (Correlation, VIF, Normality)  
✅ Feature Engineering (Temporal, Star Power, Studio, etc.)  
✅ Model Comparison & Cross-Validation  
✅ Evaluation Metrics (R², MAE, RMSE, Learning Curves)  

---
**Navigate tabs above to explore each component**
"""
        return metrics
    
    # Tab 2: Data Quality
    def quality_tab(self):
        """Embed data quality HTML report"""
        html = load_html('demo/data_quality_report.html')
        return f'<iframe srcdoc="{html}" width="100%" height="800px"></iframe>'
    
    # Tab 3: EDA
    def eda_tab(self):
        """Display EDA plots"""
        plots_dir = Path('demo/plots')
        plot_files = {
            'distributions': list((plots_dir / 'distributions').glob('*.png'))[:4],
            'relationships': list((plots_dir / 'relationships').glob('*.png'))[:4],
            'categorical': list((plots_dir / 'categorical').glob('*.png'))[:3],
            'time_series': list((plots_dir / 'time_series').glob('*.png'))[:3]
        }
        return plot_files
    
    # Tab 4: Statistics
    def stats_tab(self):
        """Display statistical analysis"""
        stats_md = f"""
## 📊 Summary Statistics
**Central Tendency & Dispersion for Key Features**
"""
        corr_desc = """
## 🔗 Correlation Analysis
**Multicollinearity Check (VIF)**
VIF > 10 indicates high multicollinearity
"""
        return stats_md, self.stats, corr_desc, self.vif
    
    # Tab 5: Prediction
    def predict(self, budget, runtime, rating, rating_count, is_franchise, has_actor, is_summer, is_studio):
        """Make prediction with all 58 features"""
        if not self.model:
            return "❌ Model not loaded", None
        
        try:
            # Build feature dict with defaults
            features = {fname: 0 for fname in self.model['feature_names']}
            
            # Fill defaults by pattern
            for fname in features:
                if 'Count' in fname: features[fname] = 3
                elif 'Year' in fname: features[fname] = 2024
                elif 'Log_Budget' in fname: features[fname] = np.log1p(budget)
                elif 'Ratio' in fname: features[fname] = 1.0
                elif 'Score' in fname or '_Gross' in fname: features[fname] = 100000000
            
            # Override with user inputs
            inputs = {
                'Budget': budget, 'Runtime': runtime, 'Rating': rating, 'Rating_Count': rating_count,
                'Log_Budget': np.log1p(budget), 'Release_Year': 2024,
                'Is_Franchise': int(is_franchise), 'Has_A_List_Actor': int(has_actor),
                'Is_Summer_Release': int(is_summer), 'Is_Major_Studio': int(is_studio)
            }
            for k, v in inputs.items():
                if k in features:
                    features[k] = v
            
            # Predict
            X = np.array([[features[f] for f in self.model['feature_names']]])
            pred = self.model['model'].predict(X)[0]
            profit = pred - budget
            roi = (profit / budget * 100) if budget > 0 else 0
            
            result = f"""
## 💰 Prediction Results
- **Predicted Gross:** ${pred:,.0f}
- **Budget:** ${budget:,.0f}
- **Profit:** ${profit:,.0f}
- **ROI:** {roi:.1f}%

*Using {self.model.get('model_name', 'Random Forest')} with {len(self.model['feature_names'])} features*
"""
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.bar(['Budget', 'Gross'], [budget, pred], color=['#FF6B6B', '#4ECDC4'])
            ax1.set_title('Budget vs Predicted Gross')
            ax1.set_ylabel('Amount ($)')
            for i, v in enumerate([budget, pred]):
                ax1.text(i, v, f'${v/1e6:.0f}M', ha='center', va='bottom')
            
            ax2.barh(['ROI'], [roi], color='#45B7D1' if roi > 0 else '#FF6B6B')
            ax2.set_xlabel('ROI (%)')
            ax2.set_title('Return on Investment')
            ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            return result, fig
        except Exception as e:
            return f"❌ Error: {e}", None
    
    # Tab 6: Model Comparison
    def models_tab(self):
        """Display all 7 models comparison"""
        if self.comparison is None:
            return "⚠️ Model comparison not available", None, None
        
        summary = f"""
## 🏆 Model Performance Comparison
**{len(self.comparison)} models trained and evaluated**

### Top 3 Models
"""
        for i, row in self.comparison.head(3).iterrows():
            medal = ['🥇', '🥈', '🥉'][i]
            summary += f"""
{medal} **{row['Model']}**
- Test R²: **{row['Test_R2']:.4f}** | MAE: ${row['Test_MAE']:,.0f}
- Train R²: {row['Train_R2']:.4f} | Overfit: {row['Overfit']:.4f}
"""
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = self.comparison['Model'].values
        test_r2 = self.comparison['Test_R2'].values
        train_r2 = self.comparison['Train_R2'].values
        
        x = np.arange(len(models))
        ax1.barh(x, test_r2, color='#4ECDC4', alpha=0.8, label='Test R²')
        ax1.set_yticks(x)
        ax1.set_yticklabels(models, fontsize=9)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Test R² Comparison')
        ax1.axvline(0.9, color='red', linestyle='--', alpha=0.3, label='0.9 threshold')
        ax1.legend()
        
        ax2.scatter(train_r2, test_r2, s=100, alpha=0.6, c=range(len(models)), cmap='viridis')
        ax2.plot([0.8, 1.0], [0.8, 1.0], 'r--', alpha=0.3, label='Perfect fit')
        ax2.set_xlabel('Train R²')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Overfitting Analysis')
        ax2.legend()
        for i, model in enumerate(models):
            ax2.annotate(model, (train_r2[i], test_r2[i]), fontsize=7, alpha=0.7)
        
        plt.tight_layout()
        
        return summary, self.comparison, fig
    
    # Tab 7: Features
    def features_tab(self):
        """Feature engineering showcase"""
        if not self.model or 'model' not in self.model:
            return "⚠️ Feature importance not available", None
        
        # Get feature importance
        if hasattr(self.model['model'], 'feature_importances_'):
            importances = self.model['model'].feature_importances_
            feat_df = pd.DataFrame({
                'Feature': self.model['feature_names'],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            summary = f"""
## 🔧 Feature Engineering
**Total Features:** {len(self.model['feature_names'])}

### Feature Categories Created
1. **Temporal** (9): Release_Month, Quarter, DayOfWeek, Is_Summer, Is_Holiday, Is_Awards_Season, Decade, Movie_Age
2. **Budget** (5): Log_Budget, Budget_Tier, Budget_Percentile, Budget_Per_Genre, Is_High_Budget
3. **Content** (10+): Genre binaries, Is_Franchise, Is_Sequel, Is_Adaptation, Has_Superhero
4. **Star Power** (6): Has_A_List_Actor, Has_A_List_Director, Top_Actor_Count, Director/Actor Avg Gross
5. **Studio** (3): Is_Major_Studio, Studio_Avg_Gross, Studio_Count
6. **Rating** (8): MPAA_Encoded, Is_Family_Friendly, Target_Audience, Rating_Bucket, Is_Highly_Rated
7. **Geographic** (6): Is_English, Is_Multilingual, Is_US_Production, Primary_Country, Market_Reach
8. **Ratios** (4): Budget_Runtime_Ratio, Cast_to_Budget_Ratio, Keyword_Density, Production_Scale

### Top 20 Most Important Features
"""
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feat_df)))
            ax.barh(range(len(feat_df)), feat_df['Importance'], color=colors)
            ax.set_yticks(range(len(feat_df)))
            ax.set_yticklabels(feat_df['Feature'], fontsize=9)
            ax.set_xlabel('Importance')
            ax.set_title('Top 20 Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
            
            return summary, feat_df, fig
        else:
            return "⚠️ Model doesn't support feature importance", None, None
    
    # Build Interface
    def build(self):
        """Create Gradio interface"""
        with gr.Blocks(title="🎬 Box Office Prediction", theme=gr.themes.Soft()) as app:
            
            with gr.Tabs():
                # Tab 1: Overview
                with gr.Tab("🏠 Overview"):
                    gr.Markdown(self.overview_tab())
                
                # Tab 2: Data Quality
                with gr.Tab("📊 Data Quality"):
                    gr.HTML(self.quality_tab())
                
                # Tab 3: EDA
                with gr.Tab("📈 EDA"):
                    gr.Markdown("## Exploratory Data Analysis\n**25+ visualizations generated**")
                    plot_files = self.eda_tab()
                    
                    gr.Markdown("### Distribution Analysis")
                    with gr.Row():
                        for img in plot_files['distributions']:
                            gr.Image(value=load_image(img), label=img.stem)
                    
                    gr.Markdown("### Relationship Analysis")
                    with gr.Row():
                        for img in plot_files['relationships']:
                            gr.Image(value=load_image(img), label=img.stem)
                    
                    gr.Markdown("### Categorical Analysis")
                    with gr.Row():
                        for img in plot_files['categorical']:
                            gr.Image(value=load_image(img), label=img.stem)
                    
                    gr.Markdown("### Time Series Analysis")
                    with gr.Row():
                        for img in plot_files['time_series']:
                            gr.Image(value=load_image(img), label=img.stem)
                
                # Tab 4: Statistics
                with gr.Tab("🔬 Statistics"):
                    stats_md, stats_df, corr_md, vif_df = self.stats_tab()
                    gr.Markdown(stats_md)
                    if stats_df is not None:
                        gr.Dataframe(stats_df, interactive=False)
                    gr.Markdown(corr_md)
                    if vif_df is not None:
                        gr.Dataframe(vif_df, interactive=False)
                
                # Tab 5: Prediction
                with gr.Tab("🎯 Predict"):
                    gr.Markdown("### Make Box Office Prediction")
                    with gr.Row():
                        with gr.Column():
                            budget = gr.Number(label="Budget ($)", value=50000000, minimum=0)
                            runtime = gr.Number(label="Runtime (min)", value=120, minimum=40, maximum=300)
                            rating = gr.Slider(label="IMDb Rating", minimum=1, maximum=10, value=7.0, step=0.1)
                            rating_count = gr.Number(label="Rating Count", value=100000, minimum=0)
                        with gr.Column():
                            is_franchise = gr.Checkbox(label="Part of Franchise", value=False)
                            has_actor = gr.Checkbox(label="Has A-List Actor", value=False)
                            is_summer = gr.Checkbox(label="Summer Release", value=False)
                            is_studio = gr.Checkbox(label="Major Studio", value=True)
                    
                    predict_btn = gr.Button("🎬 Predict", variant="primary")
                    pred_output = gr.Markdown()
                    pred_plot = gr.Plot()
                    
                    predict_btn.click(
                        self.predict,
                        [budget, runtime, rating, rating_count, is_franchise, has_actor, is_summer, is_studio],
                        [pred_output, pred_plot]
                    )
                
                # Tab 6: Models
                with gr.Tab("🏆 Models"):
                    summary, comp_df, comp_fig = self.models_tab()
                    gr.Markdown(summary)
                    if comp_df is not None:
                        gr.Dataframe(comp_df, interactive=False)
                    if comp_fig is not None:
                        gr.Plot(comp_fig)
                
                # Tab 7: Features
                with gr.Tab("📚 Features"):
                    feat_summary, feat_df, feat_fig = self.features_tab()
                    gr.Markdown(feat_summary)
                    if feat_df is not None:
                        gr.Dataframe(feat_df, interactive=False)
                    if feat_fig is not None:
                        gr.Plot(feat_fig)
        
        return app

def main():
    """Launch app"""
    print("="*60)
    print("🎬 COMPREHENSIVE BOX OFFICE PREDICTION DEMO")
    print("="*60)
    demo = BoxOfficeDemo()
    app = demo.build()
    print("\n✓ Launching at http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
