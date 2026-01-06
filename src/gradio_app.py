"""
Box Office Prediction - Interactive Demo
Course: IT4142E - Introduction to Data Science
Enhanced UI with Interactive Visualizations and Better Contrast
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

# Helper functions
def load_pickle(path):
    """Load pickle file safely"""
    try:
        return pickle.load(open(path, 'rb')) if Path(path).exists() else None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_csv(path):
    """Load CSV safely"""
    try:
        return pd.read_csv(path) if Path(path).exists() else None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_image(path):
    """Load image safely"""
    try:
        return Image.open(path) if Path(path).exists() else None
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


class BoxOfficeDemo:
    def __init__(self):
        print("📊 Loading resources...")
        self.model = load_pickle('models/box_office_model.pkl')
        self.df = load_csv('dataset/data_cleaned.csv')
        self.comparison = load_csv('demo/model_comparison.csv')
        self.stats = load_csv('demo/stats_summary.csv')
        self.vif = load_csv('demo/vif_report.csv')
        
        if self.df is not None and self.model is not None:
            print(f"✓ Loaded: {len(self.df)} movies, {len(self.model['feature_names'])} features")
        else:
            print("⚠️ Warning: Some resources failed to load")
    
    # Tab 1: Overview
    def overview_tab(self):
        """Project overview - clean and simple"""
        if self.df is None or self.model is None or self.comparison is None:
            return "⚠️ Error: Required data not loaded. Please run `python main.py` first."
        
        overview_text = f"""
## 📊 Project Overview

**Course:** IT4142E - Introduction to Data Science  
**Objective:** Predict worldwide box office gross revenue using machine learning

### Key Metrics

| Metric | Value |
|--------|-------|
| 📁 Total Movies | **{len(self.df):,}** |
| 🔧 Features Engineered | **{len(self.model['feature_names'])}** |
| 🏆 Best Model | **{self.model.get('model_name', 'Random Forest')}** |
| 📈 Test R² Score | **{self.comparison.iloc[0]['Test_R2']:.4f}** |
| 📉 Test MAE | **${self.comparison.iloc[0]['Test_MAE']/1e6:.1f}M** |
| 📅 Time Period | {self.df['Release_Year'].min():.0f} - {self.df['Release_Year'].max():.0f} |

### 📚 Data Science Pipeline

<div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6;">
<pre style="font-size: 16px; line-height: 1.8; margin: 0; font-family: 'Courier New', monospace;">
<b>Data Collection</b> → <b>Data Cleaning</b> → <b>EDA</b> → <b>Feature Engineering</b> → <b>Model Training</b> → <b>Evaluation</b>
   (Scrapy)         (Pipeline)      (25+ viz)      (52 features)         (7 models)        (Metrics)
</pre>
</div>

### 🎯 Models Compared
Linear Regression • Ridge • Lasso • ElasticNet • Decision Tree • Random Forest • Gradient Boosting

---
**Navigate the tabs above to explore each component in detail**
"""
        return overview_text
    
    # Tab 2: Data Quality
    def quality_tab(self):
        """Display data quality metrics with visual charts"""
        if self.df is None:
            return "⚠️ Data not loaded", None, None, None
        
        # Basic metrics
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        total_cells = total_rows * total_cols
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        summary = f"""
## 📊 Data Quality Report

### Dataset Summary
- **Total Records:** {total_rows:,}
- **Total Features:** {total_cols}
- **Data Completeness:** {completeness:.2f}%
- **Missing Values:** {missing_cells:,} cells
"""
        
        # Missing values by column
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing': self.df.isnull().sum().values,
            'Percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2)
        }).sort_values('Missing', ascending=False).head(10)
        
        # Visualization: Missing values bar chart
        fig_missing = go.Figure(go.Bar(
            x=missing_df['Percentage'],
            y=missing_df['Column'],
            orientation='h',
            marker=dict(color='#ef4444', opacity=0.7),
            text=missing_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>'
        ))
        
        fig_missing.update_layout(
            title="Top 10 Features with Missing Values",
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Feature",
            height=400,
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        # Numeric summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:10]
        summary_stats = self.df[numeric_cols].describe().T.round(2)
        
        return summary, missing_df, fig_missing, summary_stats
    
    # Tab 3: Enhanced EDA with Interactive Plotly
    def create_interactive_eda_plots(self):
        """Generate interactive Plotly charts for EDA"""
        if self.df is None:
            return None, None, None, None
        
        # 1. Distribution: Gross Revenue
        fig_gross = go.Figure()
        fig_gross.add_trace(go.Histogram(
            x=self.df['Gross_worldwide'],
            nbinsx=50,
            marker_color='#3b82f6',
            opacity=0.7,
            name='Gross Revenue'
        ))
        fig_gross.update_layout(
            title="Worldwide Gross Distribution",
            xaxis_title="Gross Revenue ($)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400,
            hovermode='x'
        )
        
        # 2. Relationship: Budget vs Gross
        fig_budget_gross = px.scatter(
            self.df.sample(min(1000, len(self.df))),
            x='Budget',
            y='Gross_worldwide',
            trendline='ols',
            color='Rating',
            size='Rating_Count',
            hover_data=['Movie_Title', 'Release_Year'],
            color_continuous_scale='Viridis',
            title="Budget vs Worldwide Gross (with Rating)"
        )
        fig_budget_gross.update_layout(
            template="plotly_white",
            height=500,
            xaxis_title="Budget ($)",
            yaxis_title="Worldwide Gross ($)"
        )
        
        # 3. Categorical: Genre Performance
        if 'Genre' in self.df.columns:
            # Get top genres
            genre_counts = self.df['Genre'].str.split(',').explode().str.strip().value_counts().head(10)
            genre_gross = []
            for genre in genre_counts.index:
                avg_gross = self.df[self.df['Genre'].str.contains(genre, na=False)]['Gross_worldwide'].mean()
                genre_gross.append({'Genre': genre, 'Avg_Gross': avg_gross, 'Count': genre_counts[genre]})
            
            genre_df = pd.DataFrame(genre_gross).sort_values('Avg_Gross', ascending=False)
            
            fig_genre = go.Figure()
            fig_genre.add_trace(go.Bar(
                x=genre_df['Genre'],
                y=genre_df['Avg_Gross'],
                marker_color='#8b5cf6',
                text=genre_df['Avg_Gross'].apply(lambda x: f'${x/1e6:.0f}M'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Gross: $%{y:,.0f}<extra></extra>'
            ))
            fig_genre.update_layout(
                title="Average Gross by Genre (Top 10)",
                xaxis_title="Genre",
                yaxis_title="Average Gross ($)",
                template="plotly_white",
                height=400
            )
        else:
            fig_genre = None
        
        # 4. Temporal: Release Year Trends
        yearly_stats = self.df.groupby('Release_Year').agg({
            'Gross_worldwide': ['mean', 'count'],
            'Budget': 'mean'
        }).reset_index()
        yearly_stats.columns = ['Year', 'Avg_Gross', 'Count', 'Avg_Budget']
        
        fig_temporal = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Average Gross Over Time", "Number of Movies per Year"),
            vertical_spacing=0.15
        )
        
        fig_temporal.add_trace(
            go.Scatter(
                x=yearly_stats['Year'],
                y=yearly_stats['Avg_Gross'],
                mode='lines+markers',
                marker=dict(color='#10b981', size=8),
                line=dict(width=2),
                name='Avg Gross',
                hovertemplate='<b>%{x}</b><br>Avg Gross: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig_temporal.add_trace(
            go.Bar(
                x=yearly_stats['Year'],
                y=yearly_stats['Count'],
                marker_color='#f59e0b',
                name='Movie Count',
                hovertemplate='<b>%{x}</b><br>Movies: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_temporal.update_layout(
            template="plotly_white",
            height=600,
            showlegend=False
        )
        fig_temporal.update_xaxes(title_text="Year", row=2, col=1)
        fig_temporal.update_yaxes(title_text="Avg Gross ($)", row=1, col=1)
        fig_temporal.update_yaxes(title_text="Count", row=2, col=1)
        
        return fig_gross, fig_budget_gross, fig_genre, fig_temporal
    
    # Tab 4: Statistics
    def stats_tab(self):
        """Display statistical analysis with interactive heatmap"""
        if self.df is None:
            return "⚠️ Data not loaded", None, None, None, None
        
        stats_md = """
## 📊 Statistical Analysis

### Summary Statistics
Central tendency and dispersion for key features
"""
        
        # Correlation heatmap using Plotly
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:15]
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 8, "color": "black"},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix (Interactive)",
            xaxis_title="Features",
            yaxis_title="Features",
            height=700,
            template="plotly_white"
        )
        
        vif_md = """
### Multicollinearity Check (VIF)

**Interpretation:**
- VIF < 5: ✅ Low multicollinearity  
- VIF 5-10: ⚠️ Moderate multicollinearity  
- VIF > 10: ❌ High multicollinearity
"""
        
        return stats_md, self.stats, fig, vif_md, self.vif
    
    # Tab 5: Prediction
    def predict(self, budget, runtime, rating, rating_count, genre, is_franchise, has_actor, is_summer, is_studio):
        """Make prediction with interactive Plotly charts"""
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
                'Budget': budget, 
                'Runtime': runtime, 
                'Rating': rating, 
                'Rating_Count': rating_count,
                'Log_Budget': np.log1p(budget), 
                'Release_Year': 2024,
                'Is_Franchise': int(is_franchise), 
                'Has_A_List_Actor': int(has_actor),
                'Is_Summer_Release': int(is_summer), 
                'Is_Major_Studio': int(is_studio)
            }
            
            # Genre encoding
            genre_map = {'Action': 0, 'Comedy': 1, 'Drama': 2, 'Sci-Fi': 3, 'Thriller': 4}
            if 'Primary_Genre' in features:
                features['Primary_Genre'] = genre_map.get(genre, 0)
            
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

| Metric | Value |
|--------|-------|
| **Predicted Gross** | ${pred:,.0f} |
| **Budget** | ${budget:,.0f} |
| **Profit** | ${profit:,.0f} |
| **ROI** | {roi:.1f}% |

*Model: {self.model.get('model_name', 'Random Forest')} with {len(self.model['feature_names'])} features*
"""
            
            # Enhanced Plotly visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Budget vs Predicted Gross", "Return on Investment"),
                specs=[[{"type": "bar"}, {"type": "indicator"}]],
                column_widths=[0.5, 0.5]
            )
            
            # Bar chart with better colors
            colors = ['#ef4444' if v < pred/2 else '#3b82f6' if v == budget else '#10b981' 
                     for v in [budget, pred]]
            fig.add_trace(
                go.Bar(
                    x=['Budget', 'Predicted Gross'],
                    y=[budget, pred],
                    marker_color=colors,
                    text=[f'${budget/1e6:.0f}M', f'${pred/1e6:.0f}M'],
                    textposition='outside',
                    textfont=dict(size=14, color='black'),
                    hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # ROI Indicator with color coding
            fig.add_trace(
                go.Indicator(
                    mode="number+delta+gauge",
                    value=roi,
                    delta={'reference': 100, 'relative': False, 'suffix': '%'},
                    title={'text': "ROI (%)", 'font': {'size': 18}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [-100, 500]},
                        'bar': {'color': '#10b981' if roi > 100 else '#f59e0b' if roi > 0 else '#ef4444'},
                        'steps': [
                            {'range': [-100, 0], 'color': '#fee2e2'},
                            {'range': [0, 100], 'color': '#fef3c7'},
                            {'range': [100, 500], 'color': '#d1fae5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=450,
                showlegend=False,
                template="plotly_white"
            )
            fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
            
            return result, fig
        except Exception as e:
            import traceback
            return f"❌ Error: {e}\n\n{traceback.format_exc()}", None
    
    # Tab 6: Model Comparison
    def models_tab(self):
        """Display model comparison with enhanced interactive charts"""
        if self.comparison is None:
            return "⚠️ Model comparison not available", None, None
        
        summary = f"""
## 🏆 Model Performance Comparison

**Total Models Trained:** {len(self.comparison)}

### Top 3 Models
"""
        medals = ['🥇', '🥈', '🥉']
        for i, row in self.comparison.head(3).iterrows():
            medal = medals[i] if i < 3 else '•'
            summary += f"""
{medal} **{row['Model']}**
- Test R²: **{row['Test_R2']:.4f}** | MAE: ${row['Test_MAE']:,.0f}
- Train R²: {row['Train_R2']:.4f} | Overfit: {row['Overfit']:.4f}
"""
        
        # Enhanced comparison chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Test R² Comparison", "Train vs Test R² (Overfitting Analysis)"),
            column_widths=[0.5, 0.5]
        )
        
        models = self.comparison['Model'].values
        test_r2 = self.comparison['Test_R2'].values
        train_r2 = self.comparison['Train_R2'].values
        
        # Gradient colors for bars
        colors = px.colors.sequential.Viridis[::2]  # Get every other color
        
        # Bar chart for Test R²
        fig.add_trace(
            go.Bar(
                y=models,
                x=test_r2,
                orientation='h',
                marker=dict(
                    color=test_r2,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="R²", x=0.45)
                ),
                text=test_r2.round(4),
                textposition='outside',
                textfont=dict(size=10, color='black'),
                name='Test R²',
                hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Scatter for overfitting with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=train_r2,
                y=test_r2,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=list(range(len(models))),
                    colorscale='Plasma',
                    showscale=False,
                    line=dict(width=2, color='white')
                ),
                text=models,
                textposition='top center',
                textfont=dict(size=9, color='black'),
                name='Models',
                hovertemplate='<b>%{text}</b><br>Train R²: %{x:.4f}<br>Test R²: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Perfect fit line
        fig.add_trace(
            go.Scatter(
                x=[0.8, 1.0],
                y=[0.8, 1.0],
                mode='lines',
                line=dict(dash='dash', color='red', width=2),
                name='Perfect Fit',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="R² Score", row=1, col=1)
        fig.update_xaxes(title_text="Train R²", row=1, col=2)
        fig.update_yaxes(title_text="Test R²", row=1, col=2)
        
        fig.update_layout(
            height=550,
            showlegend=False,
            template="plotly_white",
            hovermode='closest'
        )
        
        return summary, self.comparison, fig
    
    # Tab 7: Feature Importance with Detailed Explanations
    def features_tab(self):
        """Feature engineering showcase with comprehensive explanations"""
        if not self.model or 'model' not in self.model:
            return "⚠️ Feature importance not available", None, None, None
        
        # Feature Dictionary - Comprehensive explanations
        feature_dictionary = """
## 📖 Complete Feature Engineering Dictionary

---

## 🔍 Part 1: Raw Features (17 fields scraped from IMDb)

These are the original fields collected using **Scrapy + Playwright** from IMDb movie pages.

### Core Identifiers & Metadata
| Field | Description | Example | Where Scraped From |
|-------|-------------|---------|-------------------|
| **Movie_ID** | IMDb unique identifier | tt0111161 | URL slug |
| **Movie_Title** | Full movie title | "The Shawshank Redemption" | `<h1>` tag |
| **Release_Year** | Year of theatrical release | 1994 | Title metadata |
| **Release_Data** | Full release date | "1994-09-23" | Release info section |

### Financial Data
| Field | Description | Example | Data Source |
|-------|-------------|---------|-------------|
| **Budget** | Production budget (USD) | $25,000,000 | Box office section |
| **Gross_worldwide** | Total worldwide box office | $28,341,469 | Box office totals |
| **Gross_usa** | US/Canada box office | $28,341,469 | Domestic gross |

### Content Information
| Field | Description | Example | Extraction Method |
|-------|-------------|---------|------------------|
| **Genre** | Comma-separated genres | "Drama, Crime" | Genre tags |
| **Runtime** | Duration in minutes | 142 | Tech specs |
| **ListOfCertificate** | MPAA/age ratings | "R, 15, 16" | Certificates section |
| **Languages** | Audio languages | "English" | Language list |
| **Countries** | Production countries | "United States" | Country of origin |

### People & Studios
| Field | Description | Example | XPath/Selector |
|-------|-------------|---------|----------------|
| **Cast** | Main cast members (comma-sep) | "Tim Robbins, Morgan Freeman, Bob Gunton" | Cast list (top 15) |
| **Crew** | Directors & key crew | "Frank Darabont, Roger Deakins" | Credits section |
| **Studios** | Production companies | "Castle Rock Entertainment, Columbia Pictures" | Production companies |

### Engagement Metrics
| Field | Description | Example | API/Scraping |
|-------|-------------|---------|--------------|
| **Rating** | IMDb user rating (0-10) | 9.3 | Rating display |
| **Rating_Count** | Number of user ratings | 2,744,891 | Vote count |
| **Keywords** | Plot keywords | "prison, friendship, hope, escape" | Keywords section |

### Technical Notes
- **Data Collection:** Scrapy framework with Playwright for JavaScript rendering
- **Pagination:** "50 more" button clicks to load full cast/crew
- **Anti-Bot:** AUTOTHROTTLE + User-Agent rotation
- **Data Format:** JSON Lines (.jsonl) → CSV conversion
- **Missing Values:** Common in Budget (~40%), Gross (~25%), Keywords (~15%)

---

## 🔧 Part 2: Engineered Features (52+ features created)

From the 17 raw fields above, we engineered 52+ predictive features using domain knowledge.

### 🕐 Temporal Features (8 features)
| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Release_Month** | Month of release (1-12) | 5 = May, 12 = December |
| **Release_Quarter** | Quarter of release (1-4) | 2 = Q2 (Apr-Jun) |
| **Release_DayOfWeek** | Day of week (0-6) | 4 = Friday |
| **Is_Summer_Release** | Released May-August (1/0) | 1 = Summer blockbuster season |
| **Is_Holiday_Release** | Released Nov-Dec (1/0) | 1 = Holiday season |
| **Is_Awards_Season** | Released Jan-Feb (1/0) | 1 = Oscar bait timing |
| **Decade** | Release decade | 2020, 2010, 2000 |
| **Movie_Age** | Years since release | Current year - Release year |

### 💰 Budget & Financial Features (5 features)
| Feature | Description | Rationale |
|---------|-------------|-----------|
| **Log_Budget** | log₁₀(Budget) | Normalizes extreme budget ranges |
| **Budget_Tier** | Categorical tier | Micro/Low/Medium/High/Blockbuster |
| **Is_High_Budget** | Budget > $100M (1/0) | Identifies tentpole films |
| **Budget_Percentile** | Percentile rank (0-100) | Relative budget position |
| **Budget_Per_Genre** | Budget / Genre Median | Identifies over/under-budgeted films |

### 🎭 Content & Genre Features (12 features)
| Feature | Description | Usage |
|---------|-------------|-------|
| **Is_Action** | Contains "Action" genre (1/0) | Binary genre indicator |
| **Is_Comedy** | Contains "Comedy" genre (1/0) | Binary genre indicator |
| **Is_Drama** | Contains "Drama" genre (1/0) | Binary genre indicator |
| **Is_Thriller** | Contains "Thriller" genre (1/0) | Binary genre indicator |
| **Is_Horror** | Contains "Horror" genre (1/0) | Binary genre indicator |
| **Is_Animation** | Contains "Animation" genre (1/0) | Binary genre indicator |
| **Is_SciFi** | Contains "Sci-Fi/Fantasy" (1/0) | Binary genre indicator |
| **Genre_Popularity_Score** | Avg gross for primary genre | Historical genre performance |
| **Is_Sequel** | Detected sequel patterns (1/0) | II, III, Part 2, Chapter 2, etc. |
| **Is_Franchise** | Part of known franchise (1/0) | Marvel, Star Wars, Harry Potter, etc. |
| **Is_Adaptation** | Based on novel/comic (1/0) | Derived from keywords |
| **Has_Superhero** | Contains superhero elements (1/0) | Batman, Superman, Iron Man, etc. |

### ⭐ Star Power Features (6 features)
| Feature | Description | How Calculated |
|---------|-------------|----------------|
| **Top_Actor_Count** | Number of A-list actors | Count from 33-actor list |
| **Has_A_List_Actor** | Has any A-list actor (1/0) | Leonardo DiCaprio, Tom Cruise, etc. |
| **Has_A_List_Director** | Has A-list director (1/0) | Nolan, Spielberg, Cameron, etc. |
| **Director_Avg_Gross** | Director's historical avg gross | Based on past films in dataset |
| **Lead_Actor_Avg_Gross** | Lead actor's historical avg | Based on past films in dataset |
| **First_Director** | Primary director name | Extracted from Crew |

### 🏢 Studio Features (3 features)
| Feature | Description | Major Studios |
|---------|-------------|---------------|
| **Is_Major_Studio** | From major studio (1/0) | Disney, Warner Bros, Universal, etc. |
| **Studio_Avg_Gross** | Studio's historical avg gross | Track record indicator |
| **Primary_Studio** | Lead production studio | First studio listed |

### 🎫 Rating & Certification Features (7 features)
| Feature | Description | Values |
|---------|-------------|--------|
| **MPAA_Encoded** | Numeric MPAA encoding | G=0, PG=1, PG-13=2, R=3, NC-17=4 |
| **Is_Family_Friendly** | G or PG rated (1/0) | Target: children + families |
| **Is_Adult_Only** | R or NC-17 rated (1/0) | Adult audiences only |
| **Rating_Bucket** | IMDb rating category | Excel/Good/Average/Poor |
| **Is_Highly_Rated** | IMDb ≥ 7.5 (1/0) | Quality indicator |
| **Rating_Count_Log** | log₁₀(Rating Count) | Popularity measure |
| **Is_Popular** | >100k IMDb ratings (1/0) | High engagement |

### 🌍 Geographic Features (6 features)
| Feature | Description | Market Impact |
|---------|-------------|---------------|
| **Is_English** | Contains English language (1/0) | Global market access |
| **Is_Multilingual** | Multiple languages (1/0) | International appeal |
| **Is_US_Production** | Made in USA (1/0) | Hollywood production |
| **Is_International_Coproduction** | Multiple countries (1/0) | Cross-border production |
| **Market_Reach_Score** | Languages × Countries | Geographic diversity |
| **Primary_Country** | Lead production country | First country listed |

### 📊 Derived Ratio Features (4 features)
| Feature | Description | What It Measures |
|---------|-------------|------------------|
| **Budget_Runtime_Ratio** | Budget / Runtime ($/min) | Cost efficiency per minute |
| **Cast_to_Budget_Ratio** | Cast Count / (Budget/1M) | Star power efficiency |
| **Keyword_Density** | Keywords / Genres | Marketing intensity |
| **Production_Scale** | Studios × Countries | Production complexity |

---

**Total Engineered Features:** 52+ features created from 17 raw fields  
**Engineering Purpose:** Transform raw data into ML-ready predictive signals using domain knowledge
"""
        
        # Get feature importance
        if hasattr(self.model['model'], 'feature_importances_'):
            importances = self.model['model'].feature_importances_
            feat_df = pd.DataFrame({
                'Feature': self.model['feature_names'],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            summary = f"""
## 🔧 Feature Engineering Overview

**Total Features Used:** {len(self.model['feature_names'])}  
**Features Created:** 52+ engineered features from 17 raw IMDb fields

### 📈 Top 20 Most Important Features
"""
            
            # Enhanced feature importance chart
            fig = go.Figure(go.Bar(
                y=feat_df['Feature'],
                x=feat_df['Importance'],
                orientation='h',
                marker=dict(
                    color=feat_df['Importance'],
                    colorscale='Turbo',
                    showscale=True,
                    colorbar=dict(title="Importance"),
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
                text=feat_df['Importance'].round(4),
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Top 20 Feature Importance (Interactive - Hover for Details)",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=700,
                template="plotly_white",
                yaxis=dict(autorange="reversed")
            )
            
            return summary, feat_df, fig, feature_dictionary
        else:
            return "⚠️ Model doesn't support feature importance", None, None
    
    # Build Interface
    def build(self):
        """Create Gradio interface with clean modern theme"""
        
        with gr.Blocks(title="Box Office Prediction - IT4142E") as app:
            
            gr.Markdown("# 🎬 Box Office Revenue Prediction System\n**IT4142E - Introduction to Data Science**")
            
            with gr.Tabs():
                # Tab 1: Overview
                with gr.Tab("📊 Overview"):
                    overview_text = self.overview_tab()
                    gr.Markdown(overview_text)
                
                # Tab 2: Data Quality
                with gr.Tab("📋 Data Quality"):
                    summary, missing_df, fig_missing, stats_df = self.quality_tab()
                    gr.Markdown(summary)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Missing Values Chart")
                            if fig_missing is not None:
                                gr.Plot(fig_missing)
                        with gr.Column():
                            gr.Markdown("### Missing Values Table")
                            if missing_df is not None:
                                gr.Dataframe(missing_df, interactive=False)
                    
                    gr.Markdown("### Numeric Features Summary")
                    if stats_df is not None:
                        gr.Dataframe(stats_df, interactive=False)
                
                # Tab 3: EDA - ENHANCED with Interactive Plotly
                with gr.Tab("📈 Exploratory Analysis"):
                    gr.Markdown("## 📈 Exploratory Data Analysis\n**Interactive visualizations - zoom, pan, and hover for details**")
                    
                    fig_gross, fig_budget_gross, fig_genre, fig_temporal = self.create_interactive_eda_plots()
                    
                    with gr.Row():
                        with gr.Column():
                            if fig_gross:
                                gr.Plot(fig_gross)
                        with gr.Column():
                            if fig_budget_gross:
                                gr.Plot(fig_budget_gross)
                    
                    with gr.Row():
                        if fig_genre:
                            gr.Plot(fig_genre)
                    
                    with gr.Row():
                        if fig_temporal:
                            gr.Plot(fig_temporal)
                
                # Tab 4: Statistics
                with gr.Tab("🔬 Statistical Analysis"):
                    stats_md, stats_df, corr_fig, vif_md, vif_df = self.stats_tab()
                    gr.Markdown(stats_md)
                    if stats_df is not None:
                        gr.Dataframe(stats_df, interactive=False)
                    
                    gr.Markdown("### Interactive Correlation Matrix")
                    if corr_fig is not None:
                        gr.Plot(corr_fig)
                    
                    gr.Markdown(vif_md)
                    if vif_df is not None:
                        gr.Dataframe(vif_df, interactive=False)
                
                # Tab 5: Prediction
                with gr.Tab("🎯 Make Prediction"):
                    gr.Markdown("### Predict Box Office Revenue")
                    
                    with gr.Row():
                        with gr.Column():
                            budget = gr.Number(label="💵 Budget ($)", value=50000000, minimum=0)
                            runtime = gr.Number(label="⏱️ Runtime (minutes)", value=120, minimum=40, maximum=300)
                            rating = gr.Slider(label="⭐ IMDb Rating", minimum=1, maximum=10, value=7.0, step=0.1)
                            rating_count = gr.Number(label="👥 Rating Count", value=100000, minimum=0)
                        
                        with gr.Column():
                            genre = gr.Dropdown(
                                label="🎭 Primary Genre",
                                choices=["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"],
                                value="Action"
                            )
                            is_franchise = gr.Checkbox(label="🎬 Part of Franchise", value=False)
                            has_actor = gr.Checkbox(label="⭐ Has A-List Actor", value=False)
                            is_summer = gr.Checkbox(label="☀️ Summer Release (Jun-Aug)", value=False)
                            is_studio = gr.Checkbox(label="🏢 Major Studio", value=True)
                    
                    predict_btn = gr.Button("🎬 Predict Box Office Revenue", variant="primary", size="lg")
                    pred_output = gr.Markdown()
                    pred_plot = gr.Plot()
                    
                    predict_btn.click(
                        self.predict,
                        [budget, runtime, rating, rating_count, genre, is_franchise, has_actor, is_summer, is_studio],
                        [pred_output, pred_plot]
                    )
                
                # Tab 6: Models
                with gr.Tab("🏆 Model Comparison"):
                    summary, comp_df, comp_fig = self.models_tab()
                    gr.Markdown(summary)
                    
                    gr.Markdown("### Full Comparison Table")
                    if comp_df is not None:
                        gr.Dataframe(comp_df, interactive=False)
                    
                    if comp_fig is not None:
                        gr.Plot(comp_fig)
                
                # Tab 7: Features
                with gr.Tab("📚 Feature Engineering"):
                    feat_summary, feat_df, feat_fig, feat_dict = self.features_tab()
                    gr.Markdown(feat_summary)
                    
                    if feat_df is not None:
                        gr.Dataframe(feat_df, interactive=False)
                    
                    if feat_fig is not None:
                        gr.Plot(feat_fig)
                    
                    # Feature Dictionary
                    if feat_dict:
                        gr.Markdown(feat_dict)
            
            gr.Markdown("""
---
**Project:** Box Office Revenue Prediction | **Course:** IT4142E - Introduction to Data Science  
**Pipeline:** Data Collection (Scrapy) → Cleaning → EDA → Feature Engineering (52 features) → ML (7 models) → Evaluation
""")
        
        return app


def main():
    """Launch application"""
    print("="*70)
    print("🎬 BOX OFFICE PREDICTION - INTERACTIVE DEMO")
    print("   IT4142E - Introduction to Data Science")
    print("="*70)
    
    demo = BoxOfficeDemo()
    app = demo.build()
    
    print("\n✓ Demo ready!")
    print("📍 Local URL: http://localhost:7860")
    print("📊 Features: Interactive Plotly charts with zoom, pan, and hover")
    print("🎨 Enhanced UI with better contrast and visual appeal")
    print("\n" + "="*70 + "\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        # Gradio 6.0: Simple CSS that works with both dark and light modes
        css="""
        /* Simple professional CSS that works with both themes */
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif !important;
        }
        .tab-nav button {
            font-size: 15px !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            padding: 10px 16px !important;
            margin: 0 4px !important;
        }
        h1 {
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 12px;
            font-size: 2.2em !important;
            font-weight: 700 !important;
        }
        h2 {
            margin-top: 24px !important;
            font-weight: 600 !important;
            font-size: 1.6em !important;
        }
        h3 {
            font-weight: 600 !important;
            font-size: 1.2em !important;
        }
        .table-wrap table {
            font-size: 14px !important;
        }
        .table-wrap th {
            font-weight: 700 !important;
        }
        pre {
            font-size: 16px !important;
            line-height: 1.6 !important;
            padding: 16px !important;
            border-radius: 6px !important;
        }
        button {
            font-weight: 600 !important;
            border-radius: 6px !important;
        }
        """
    )


if __name__ == "__main__":
    main()
