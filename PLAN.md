# 🎯 Movie Box Office Prediction - Complete Project Plan

> **Course:** IT4142E - Introduction to Data Science  
> **Objective:** Comprehensive data science project following the full DS lifecycle  
> **Note:** This plan excludes Report & Presentation phases

---

## 📋 Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Phase 1: Data Quality & Advanced Cleaning](#phase-1-data-quality-validation--advanced-cleaning)
3. [Phase 2: Feature Engineering (Deep)](#phase-2-feature-engineering-deep)
4. [Phase 3: Exploratory Data Analysis (Enhanced)](#phase-3-exploratory-data-analysis-enhanced)
5. [Phase 4: Machine Learning & Modeling](#phase-4-machine-learning--modeling)
6. [Phase 5: Evaluation & Validation](#phase-5-evaluation--validation)
7. [Phase 6: Demo Application Enhancement](#phase-6-demo-application-enhancement)
8. [Implementation Timeline](#implementation-timeline)
9. [File Structure](#file-structure)

---

## 1. Current State Analysis

### ✅ What You Already Have

| Component | Status | Description |
|-----------|--------|-------------|
| **Dataset** | ✅ Done | **19,000+ movies** crawled (100 sample in repo due to GitHub limits) |
| **Data Crawler** | ✅ Done | Scrapy + Playwright crawler for IMDb |
| **Data Joining** | ✅ Done | `join_data.py` - merges data sources |
| **Basic Cleaning** | ✅ Done | Money parsing, runtime extraction, date parsing |
| **Basic Features** | ✅ Done | Count features (Cast, Crew, Genre, etc.), Primary_Genre, ROI |
| **Model** | ✅ Done | Random Forest Regressor (12 features) |
| **EDA Plots** | ✅ Done | 6 basic visualizations |
| **Gradio App** | ✅ Done | 4-tab demo interface |

### 🔴 What's Missing / Needs Improvement

| Component | Issue | Course Requirement |
|-----------|-------|-------------------|
| **Feature Engineering** | Shallow - only counting | Need derived features, encodings |
| **EDA** | Basic plots only | Need statistical analysis, advanced viz |
| **Model Selection** | Single model, no tuning | Need comparison, cross-validation |
| **Evaluation** | Basic metrics only | Need residual analysis, learning curves |
| **Demo** | Static, limited interactivity | Need live insights, comparisons |

---

## Phase 1: Data Quality Validation & Advanced Cleaning

### 1.1 Data Quality Validation

Before processing the full 19k dataset, validate data quality:

- [ ] **1.0.1** Create `src/data_quality_report.py` to check for:
  - Missing values per column
  - Duplicate Movie_IDs
  - Invalid data types
  - Unrealistic values (negative budgets, future dates)
- [ ] **1.0.2** Generate data quality report (`demo/data_quality_report.html`)

### 1.2 Enhanced Data Cleaning Pipeline

Extend `src/pipeline.py` with comprehensive cleaning:

### 1.3 Tasks

- [ ] **1.3.1** Handle currency conversion (non-USD budgets)
- [ ] **1.3.2** Normalize studio names (e.g., "Warner Bros" vs "Warner Bros.")
- [ ] **1.3.3** Standardize genre names (case, spelling)
- [ ] **1.3.4** Handle missing values with imputation strategies:
  - Budget: Median imputation by genre
  - Runtime: Median imputation by genre
  - Rating: Flag as separate binary feature + median fill
- [ ] **1.3.5** Detect and handle outliers:
  - Budget: IQR method, cap at 99th percentile
  - Gross: Log transformation consideration
- [ ] **1.3.6** Validate release dates (filter out future dates)

### 1.4 New Cleaning Functions

Add to `src/pipeline.py`:

```python
class BoxOfficePipeline:
    # NEW METHODS TO ADD:
    
    def impute_missing_by_genre(self, column):
        """Impute missing values using median per genre"""
        pass
    
    def normalize_studio_names(self):
        """Standardize studio names for consistency"""
        # Map variations: "Warner Bros", "Warner Bros.", "WB" → "Warner Bros."
        pass
    
    def handle_outliers(self, column, method='iqr'):
        """Cap outliers using IQR or percentile method"""
        pass
    
    def convert_currencies(self):
        """Convert non-USD budgets to USD (simplified)"""
        pass
```

### 1.5 Outlier Detection Strategy

| Column | Method | Action |
|--------|--------|--------|
| Budget | IQR (1.5x) | Cap at upper bound |
| Gross_worldwide | Log transform | Keep but flag |
| Runtime | Hard limits | 40-300 minutes |
| Rating | Valid range | 1.0-10.0 only |

---

## Phase 2: Feature Engineering (Deep)

### 2.1 Overview

**This is the most critical phase for improving model performance!**

Transform raw data into predictive features using domain knowledge.

### 2.2 New Feature Categories

#### A. Temporal Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Release_Month` | 1-12 | Summer/holiday releases perform differently |
| `Release_Quarter` | Q1-Q4 | Seasonal patterns |
| `Release_DayOfWeek` | 0-6 | Weekend releases common for blockbusters |
| `Is_Summer_Release` | Boolean | May-Aug = summer blockbuster season |
| `Is_Holiday_Release` | Boolean | Nov-Dec = holiday season |
| `Is_Awards_Season` | Boolean | Jan-Feb = Oscar bait period |
| `Decade` | 1990s, 2000s, etc. | Era-specific trends |
| `Movie_Age` | Years since release | For historical analysis |

#### B. Budget & Financial Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Budget_Tier` | Low/Medium/High/Blockbuster | Categorical budget level |
| `Log_Budget` | log10(Budget) | Normalize skewed distribution |
| `Budget_Per_Genre` | Relative to genre median | Over/under-budgeted for genre |
| `Is_High_Budget` | Boolean (>$100M) | Blockbuster indicator |
| `Budget_Percentile` | 0-100 | Relative positioning |

Budget Tier Definitions:
- **Micro**: < $5M
- **Low**: $5M - $25M  
- **Medium**: $25M - $75M
- **High**: $75M - $150M
- **Blockbuster**: > $150M

#### C. Content Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Is_Action` | Boolean | Action movies have distinct patterns |
| `Is_Comedy` | Boolean | Comedy specific patterns |
| `Is_Drama` | Boolean | Drama specific patterns |
| `Is_SciFi` | Boolean | Sci-Fi/Fantasy tend to have higher budgets |
| `Is_Horror` | Boolean | Horror often has high ROI |
| `Is_Animation` | Boolean | Animation has family audience |
| `Is_Sequel` | Boolean | Detect "2", "Part", "Chapter" in title |
| `Is_Franchise` | Boolean | Known franchises (Marvel, DC, Star Wars, etc.) |
| `Is_Adaptation` | Boolean | "based on novel/comic" in keywords |
| `Has_Superhero` | Boolean | Superhero keyword detection |
| `Genre_Popularity_Score` | Float | Historical avg gross per genre |
| `Genre_Combination_Hash` | Encoded | Multi-genre combinations |

#### D. Star Power Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Has_A_List_Actor` | Boolean | Top 100 actors by historical gross |
| `Has_A_List_Director` | Boolean | Top 50 directors by historical gross |
| `Top_Actor_Count` | 0-5 | How many A-listers in cast |
| `Director_Avg_Gross` | Float | Director's historical average |
| `Lead_Actor_Avg_Gross` | Float | First billed actor's average |
| `Cast_Experience_Score` | Float | Sum of cast's previous credits |

Create A-List lookup tables:
```python
A_LIST_ACTORS = [
    "Leonardo DiCaprio", "Tom Cruise", "Dwayne Johnson", 
    "Scarlett Johansson", "Robert Downey Jr.", ...
]
A_LIST_DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "Martin Scorsese",
    "Denis Villeneuve", "James Cameron", ...
]
```

#### E. Studio Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Is_Major_Studio` | Boolean | Disney, Warner, Universal, etc. |
| `Studio_Avg_Gross` | Float | Historical studio performance |
| `Studio_Genre_Specialty` | Float | Studio's success rate in genre |
| `Studio_Count` | Int | Number of production companies |

Major Studios List:
```python
MAJOR_STUDIOS = [
    "Walt Disney", "Warner Bros", "Universal", "Paramount",
    "Sony", "Columbia", "20th Century", "Marvel", "Lionsgate"
]
```

#### F. Rating & Certification Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `MPAA_Encoded` | Categorical | G=0, PG=1, PG-13=2, R=3, NC-17=4 |
| `Is_Family_Friendly` | Boolean | G or PG |
| `Is_Adult_Only` | Boolean | R or NC-17 |
| `Target_Audience` | Categorical | Family/Teen/Adult |
| `Rating_Bucket` | Categorical | Excellent(>8)/Good(7-8)/Average(5-7)/Poor(<5) |
| `Is_Highly_Rated` | Boolean | Rating >= 7.5 |
| `Rating_Count_Log` | Float | log10(Rating_Count) |
| `Is_Popular` | Boolean | Rating_Count > 100,000 |

#### G. Language & Geographic Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Is_English` | Boolean | Primary language check |
| `Is_Multilingual` | Boolean | Languages_Count > 1 |
| `Is_US_Production` | Boolean | USA in Countries |
| `Is_International_Coproduction` | Boolean | Countries_Count > 1 |
| `Primary_Country_Encoded` | Categorical | Top 10 countries encoded |
| `Market_Reach_Score` | Float | Languages × Countries |

#### H. Derived Ratio Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Budget_Runtime_Ratio` | Budget / Runtime | Cost per minute |
| `Cast_to_Budget_Ratio` | Cast_Count / Budget | Star efficiency |
| `Keyword_Density` | Keywords_Count / Genre_Count | Marketing intensity |
| `Production_Scale` | Studios × Countries | Production scope |

### 2.3 Implementation Tasks

- [ ] **2.3.1** Create `src/feature_engineering.py` module
- [ ] **2.3.2** Implement temporal feature extraction
- [ ] **2.3.3** Implement budget tier categorization
- [ ] **2.3.4** Implement genre binary encoding
- [ ] **2.3.5** Create A-List actor/director lookup and scoring
- [ ] **2.3.6** Implement studio reputation scoring
- [ ] **2.3.7** Implement MPAA certification encoding
- [ ] **2.3.8** Create sequel/franchise detection logic
- [ ] **2.3.9** Implement all ratio features
- [ ] **2.3.10** Create feature importance ranking after training
- [ ] **2.3.11** Save feature metadata for Gradio app

### 2.4 Feature Engineering Module Structure

Create `src/feature_engineering.py`:

```python
"""
Advanced Feature Engineering for Box Office Prediction
Transforms raw movie data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import List, Dict

# Lookup tables
A_LIST_ACTORS = [...] 
A_LIST_DIRECTORS = [...]
MAJOR_STUDIOS = [...]
FRANCHISE_KEYWORDS = ['marvel', 'dc', 'star wars', 'harry potter', 'fast furious', ...]

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_metadata = {}
    
    # --- TEMPORAL FEATURES ---
    def extract_temporal_features(self) -> 'FeatureEngineer':
        """Extract all time-based features"""
        pass
    
    def _is_summer_release(self, month: int) -> bool:
        return month in [5, 6, 7, 8]
    
    def _is_holiday_release(self, month: int) -> bool:
        return month in [11, 12]
    
    # --- BUDGET FEATURES ---
    def create_budget_features(self) -> 'FeatureEngineer':
        """Create budget tiers and log transforms"""
        pass
    
    def _categorize_budget(self, budget: float) -> str:
        if budget < 5_000_000: return 'Micro'
        elif budget < 25_000_000: return 'Low'
        elif budget < 75_000_000: return 'Medium'
        elif budget < 150_000_000: return 'High'
        else: return 'Blockbuster'
    
    # --- CONTENT FEATURES ---
    def create_genre_features(self) -> 'FeatureEngineer':
        """Binary encode top genres + genre popularity score"""
        pass
    
    def detect_sequel_franchise(self) -> 'FeatureEngineer':
        """Detect sequels and known franchises"""
        pass
    
    # --- STAR POWER FEATURES ---
    def calculate_star_power(self) -> 'FeatureEngineer':
        """Score movies by cast/director star power"""
        pass
    
    def _count_a_list_actors(self, cast_str: str) -> int:
        if pd.isna(cast_str): return 0
        cast_list = [c.strip() for c in cast_str.split(',')]
        return sum(1 for actor in cast_list if actor in A_LIST_ACTORS)
    
    # --- STUDIO FEATURES ---
    def create_studio_features(self) -> 'FeatureEngineer':
        """Studio reputation scoring"""
        pass
    
    # --- RATING FEATURES ---
    def create_rating_features(self) -> 'FeatureEngineer':
        """MPAA encoding and rating buckets"""
        pass
    
    # --- GEOGRAPHIC FEATURES ---
    def create_geographic_features(self) -> 'FeatureEngineer':
        """Language and country features"""
        pass
    
    # --- RATIO FEATURES ---
    def create_ratio_features(self) -> 'FeatureEngineer':
        """Derived ratio features"""
        pass
    
    # --- MAIN PIPELINE ---
    def run_full_engineering(self) -> pd.DataFrame:
        """Execute all feature engineering steps"""
        return (self
            .extract_temporal_features()
            .create_budget_features()
            .create_genre_features()
            .detect_sequel_franchise()
            .calculate_star_power()
            .create_studio_features()
            .create_rating_features()
            .create_geographic_features()
            .create_ratio_features()
            .df)
    
    def get_feature_list(self) -> List[str]:
        """Return list of all engineered feature names"""
        pass
    
    def get_feature_metadata(self) -> Dict:
        """Return metadata about each feature for documentation"""
        pass
```

### 2.5 Expected New Features Count

| Category | Features | Running Total |
|----------|----------|---------------|
| Original | 12 | 12 |
| Temporal | 8 | 20 |
| Budget | 5 | 25 |
| Content/Genre | 12 | 37 |
| Star Power | 6 | 43 |
| Studio | 4 | 47 |
| Rating/Certification | 7 | 54 |
| Geographic | 6 | 60 |
| Ratios | 4 | 64 |

**Target: ~40-50 final features after feature selection**

---

## Phase 3: Exploratory Data Analysis (Enhanced)

### 3.1 Statistical Analysis

Create `src/statistical_analysis.py`:

- [ ] **3.1.1** Compute summary statistics table (mean, median, std, skewness, kurtosis)
- [ ] **3.1.2** Normality tests (Shapiro-Wilk for key variables)
- [ ] **3.1.3** Correlation matrix with p-values
- [ ] **3.1.4** Multicollinearity check (VIF - Variance Inflation Factor)
- [ ] **3.1.5** Outlier analysis report (IQR, Z-score methods)

### 3.2 Enhanced Visualizations

Extend `src/eda_visualizations.py` with:

#### A. Distribution Analysis

- [ ] **3.2.1** Histogram + KDE overlay for continuous variables
- [ ] **3.2.2** Log-scale distribution plots (Budget, Gross)
- [ ] **3.2.3** Q-Q plots for normality assessment
- [ ] **3.2.4** Box plots with individual points overlay

#### B. Relationship Analysis

- [ ] **3.2.5** Scatter matrix (pairplot) for top features
- [ ] **3.2.6** Regression plots with confidence intervals
- [ ] **3.2.7** Residual plots preview

#### C. Categorical Analysis

- [ ] **3.2.8** Genre performance comparison (violin plots)
- [ ] **3.2.9** Budget tier vs Gross (grouped bar chart)
- [ ] **3.2.10** MPAA rating impact analysis
- [ ] **3.2.11** Studio performance comparison
- [ ] **3.2.12** Seasonal performance patterns

#### D. Time Series Analysis

- [ ] **3.2.13** Yearly trends (Average budget, gross over time)
- [ ] **3.2.14** Monthly release patterns
- [ ] **3.2.15** Genre popularity evolution over decades
- [ ] **3.2.16** ROI trends over time

#### E. Advanced Visualizations

- [ ] **3.2.17** Correlation heatmap with hierarchical clustering
- [ ] **3.2.18** Parallel coordinates plot for high-dimensional data
- [ ] **3.2.19** PCA biplot (2D projection of features)
- [ ] **3.2.20** Feature importance bar chart (after model training)

### 3.3 New Visualization Functions

Add to `src/eda_visualizations.py`:

```python
class EDAVisualizer:
    # NEW METHODS:
    
    def plot_distribution_with_kde(self, column, log_scale=False):
        """Histogram with kernel density estimate overlay"""
        pass
    
    def plot_qq(self, column):
        """Q-Q plot for normality assessment"""
        pass
    
    def plot_pairwise_scatter(self, columns):
        """Scatter matrix for multiple variables"""
        pass
    
    def plot_categorical_comparison(self, cat_col, num_col):
        """Violin/box plot comparing categories"""
        pass
    
    def plot_time_trends(self):
        """Multi-line plot of trends over years"""
        pass
    
    def plot_correlation_clustered(self):
        """Hierarchically clustered correlation heatmap"""
        pass
    
    def plot_parallel_coordinates(self, columns):
        """Parallel coordinates for multivariate visualization"""
        pass
    
    def plot_pca_biplot(self, n_components=2):
        """PCA projection with feature loadings"""
        pass
    
    def plot_seasonal_patterns(self):
        """Monthly/quarterly performance patterns"""
        pass
```

### 3.4 EDA Report Generation

Create `src/eda_report.py`:

- [ ] **3.4.1** Auto-generate comprehensive EDA report (HTML or PDF)
- [ ] **3.4.2** Include all plots + statistical tests
- [ ] **3.4.3** Add key insights text summaries
- [ ] **3.4.4** Save to `demo/eda_report.html`

### 3.5 Expected EDA Outputs

| Output | Location |
|--------|----------|
| Distribution plots (8+) | `demo/plots/distributions/` |
| Relationship plots (6+) | `demo/plots/relationships/` |
| Categorical plots (5+) | `demo/plots/categorical/` |
| Time series plots (4+) | `demo/plots/temporal/` |
| Advanced plots (4+) | `demo/plots/advanced/` |
| Summary statistics | `demo/stats_summary.csv` |
| Correlation matrix | `demo/correlation_matrix.csv` |
| EDA Report | `demo/eda_report.html` |

---

## Phase 4: Machine Learning & Modeling

### 4.1 Model Selection & Comparison

Create `src/model_training.py`:

#### A. Models to Compare

| Model | Sklearn Class | Hyperparameters to Tune |
|-------|---------------|-------------------------|
| Linear Regression | `LinearRegression` | None (baseline) |
| Ridge Regression | `Ridge` | alpha |
| Lasso Regression | `Lasso` | alpha |
| ElasticNet | `ElasticNet` | alpha, l1_ratio |
| Decision Tree | `DecisionTreeRegressor` | max_depth, min_samples_split |
| **Random Forest** | `RandomForestRegressor` | n_estimators, max_depth, min_samples |
| Gradient Boosting | `GradientBoostingRegressor` | n_estimators, learning_rate, max_depth |
| XGBoost (optional) | `XGBRegressor` | Various |

### 4.2 Tasks

- [ ] **4.2.1** Implement model comparison framework
- [ ] **4.2.2** Set up train/validation/test split (70/15/15)
- [ ] **4.2.3** Implement k-fold cross-validation (k=5)
- [ ] **4.2.4** Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] **4.2.5** Train all models and compare performance
- [ ] **4.2.6** Feature selection using RFE or feature importance
- [ ] **4.2.7** Save best model with metadata
- [ ] **4.2.8** Generate model comparison report

### 4.3 Model Training Module

Create `src/model_training.py`:

```python
"""
Model Training and Selection for Box Office Prediction
Compares multiple algorithms and selects the best performer
"""

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.models = {}
        self.results = {}
        self.best_model = None
    
    def define_models(self):
        """Define all models with hyperparameter grids"""
        self.models = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'Lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'Decision_Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
            },
            'Random_Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        return self
    
    def cross_validate_all(self, cv=5):
        """Perform cross-validation for all models"""
        pass
    
    def tune_hyperparameters(self, model_name):
        """GridSearchCV for specific model"""
        pass
    
    def train_best_model(self):
        """Train final model with best hyperparameters"""
        pass
    
    def evaluate_model(self, model, X, y):
        """Calculate all evaluation metrics"""
        preds = model.predict(X)
        return {
            'MAE': mean_absolute_error(y, preds),
            'RMSE': np.sqrt(mean_squared_error(y, preds)),
            'R2': r2_score(y, preds),
            'MAPE': np.mean(np.abs((y - preds) / y)) * 100
        }
    
    def generate_comparison_report(self):
        """Create model comparison DataFrame"""
        pass
    
    def save_model(self, path):
        """Save best model with metadata"""
        pass
```

### 4.4 Hyperparameter Tuning Strategy

```python
# Example: Random Forest tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
```

### 4.5 Feature Selection

- [ ] **4.5.1** Use feature importance from Random Forest
- [ ] **4.5.2** Implement Recursive Feature Elimination (RFE)
- [ ] **4.5.3** Remove features with correlation > 0.9 (multicollinearity)
- [ ] **4.5.4** Select top 30-40 features for final model

---

## Phase 5: Evaluation & Validation

### 5.1 Evaluation Metrics

Create `src/model_evaluation.py`:

| Metric | Description | Target |
|--------|-------------|--------|
| **R² Score** | Variance explained | > 0.70 |
| **MAE** | Average error | < $50M |
| **RMSE** | Penalizes large errors | < $100M |
| **MAPE** | Percentage error | < 30% |

### 5.2 Tasks

- [ ] **5.2.1** Generate comprehensive metrics for train/test sets
- [ ] **5.2.2** Create residual analysis plots
- [ ] **5.2.3** Generate learning curves (train size vs performance)
- [ ] **5.2.4** Create prediction vs actual scatter plot
- [ ] **5.2.5** Analyze prediction errors by category (genre, budget tier)
- [ ] **5.2.6** Generate model performance report

### 5.3 Evaluation Module

Create `src/model_evaluation.py`:

```python
"""
Model Evaluation and Validation
Comprehensive analysis of model performance
"""

class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
    
    def calculate_all_metrics(self):
        """Compute MAE, RMSE, R2, MAPE for both sets"""
        pass
    
    def plot_residuals(self):
        """Residual plot and histogram"""
        pass
    
    def plot_prediction_vs_actual(self):
        """Scatter plot with perfect prediction line"""
        pass
    
    def plot_learning_curve(self):
        """Learning curve (training size vs score)"""
        pass
    
    def plot_validation_curve(self, param_name, param_range):
        """Validation curve for hyperparameter"""
        pass
    
    def analyze_errors_by_category(self, df, category_col):
        """Error analysis grouped by category"""
        pass
    
    def generate_evaluation_report(self):
        """Full evaluation report with all metrics and plots"""
        pass
```

### 5.4 Expected Evaluation Outputs

| Output | Location |
|--------|----------|
| Metrics summary | `demo/model_metrics.json` |
| Residual plots | `demo/plots/evaluation/residuals.png` |
| Prediction vs Actual | `demo/plots/evaluation/pred_vs_actual.png` |
| Learning curve | `demo/plots/evaluation/learning_curve.png` |
| Error by genre | `demo/plots/evaluation/error_by_genre.png` |
| Model comparison | `demo/model_comparison.csv` |

---

## Phase 6: Demo Application Enhancement

### 6.1 Enhanced Gradio Interface

Update `src/gradio_app.py`:

### 6.2 Tasks

- [ ] **6.2.1** Add model comparison tab
- [ ] **6.2.2** Add feature importance visualization
- [ ] **6.2.3** Add interactive EDA tab with dropdown selectors
- [ ] **6.2.4** Add prediction explanation (which features contributed most)
- [ ] **6.2.5** Add movie similarity finder (nearest neighbors)
- [ ] **6.2.6** Add historical trend exploration
- [ ] **6.2.7** Improve UI/UX with custom CSS
- [ ] **6.2.8** Add export functionality (download predictions as CSV)
- [ ] **6.2.9** Add batch prediction mode (upload CSV)

### 6.3 New Gradio Tabs

| Tab | Features |
|-----|----------|
| **🎯 Predict** | Enhanced input form, prediction explanation, confidence intervals |
| **📊 EDA Explorer** | Interactive plot selector, filter by genre/year/budget |
| **📈 Model Insights** | Model comparison, feature importance, learning curves |
| **🔍 Movie Finder** | Find similar movies, compare predictions |
| **📁 Batch Mode** | Upload CSV, get batch predictions |
| **ℹ️ About** | Dataset info, methodology, model details |

### 6.4 Enhanced Gradio App Structure

```python
"""
Enhanced Gradio Application with Full DS Pipeline Integration
"""

class EnhancedBoxOfficeApp:
    def __init__(self):
        self.models = {}  # All trained models
        self.feature_engineer = None
        self.evaluator = None
        
    def predict_with_explanation(self, **kwargs):
        """Make prediction and explain feature contributions"""
        # Use SHAP or feature importance for explanation
        pass
    
    def find_similar_movies(self, movie_features, top_k=5):
        """Find nearest neighbor movies"""
        pass
    
    def interactive_eda(self, x_var, y_var, color_by, filter_genre):
        """Dynamic EDA plot generation"""
        pass
    
    def batch_predict(self, csv_file):
        """Process uploaded CSV for batch predictions"""
        pass
    
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
            # ... 6 tabs as described above
            pass
        return app
```

### 6.5 UI Improvements

```python
custom_css = """
    .gradio-container { max-width: 1200px !important; }
    .output-markdown { font-size: 1.1em; }
    .prediction-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
    }
    /* More custom styles */
"""
```

---

## Implementation Timeline

> **Note:** Data crawling already completed (19k movies available). Timeline starts from cleaning.

### Week 1: Data Quality & Feature Engineering

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Data quality validation | `data_quality_report.py` |
| 2 | Advanced data cleaning | Updated `pipeline.py` |
| 3-4 | Temporal + Budget + Genre features | `feature_engineering.py` (partial) |
| 5 | Star Power + Studio + Rating features | Complete feature module |

### Week 2: EDA & Analysis

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Statistical analysis | `statistical_analysis.py` |
| 3-4 | Enhanced visualizations (20+ plots) | Updated `eda_visualizations.py` |
| 5 | EDA report generation | `demo/eda_report.html` |

### Week 3: Modeling & Evaluation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Model comparison framework (5+ models) | `model_training.py` |
| 3 | Hyperparameter tuning (GridSearchCV) | Best model trained |
| 4-5 | Evaluation & validation | `model_evaluation.py` |

### Week 4: Demo Application

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Enhanced Gradio app (6 tabs) | Updated `gradio_app.py` |
| 3-4 | Interactive EDA + batch prediction | Complete demo app |
| 5 | Testing, polish & final integration | Final demo ready |

---

## File Structure

### Final Project Structure

```
movie-earnings-ds/
├── main.py                              # Main pipeline runner
├── requirements.txt                     # Dependencies
├── PLAN.md                              # This file
├── README.md                            # Project documentation
│
├── dataset/
│   ├── data.csv                         # Raw crawled data
│   ├── data_joined.csv                  # Merged data
│   ├── data_cleaned.csv                 # Cleaned data
│   └── data_featured.csv                # Feature-engineered data (NEW)
│
├── models/
│   ├── box_office_model.pkl             # Best model
│   ├── model_comparison.json            # All model results (NEW)
│   └── feature_metadata.json            # Feature info (NEW)
│
├── demo/
│   ├── plots/
│   │   ├── distributions/               # Distribution plots (NEW)
│   │   ├── relationships/               # Relationship plots (NEW)
│   │   ├── categorical/                 # Category comparisons (NEW)
│   │   ├── temporal/                    # Time trends (NEW)
│   │   ├── advanced/                    # PCA, parallel coords (NEW)
│   │   └── evaluation/                  # Model evaluation plots (NEW)
│   ├── eda_report.html                  # Full EDA report (NEW)
│   ├── model_metrics.json               # Metrics summary (NEW)
│   └── model_comparison.csv             # Model comparison table (NEW)
│
└── src/
    ├── __init__.py
    ├── pipeline.py                      # Enhanced cleaning pipeline
    ├── feature_engineering.py           # NEW: Feature engineering module
    ├── statistical_analysis.py          # NEW: Stats tests
    ├── eda_visualizations.py            # Enhanced EDA module
    ├── eda_report.py                    # NEW: Report generator
    ├── model_training.py                # NEW: Model comparison
    ├── model_evaluation.py              # NEW: Evaluation module
    ├── gradio_app.py                    # Enhanced demo app
    ├── data_quality_report.py           # NEW: Data quality checks
    ├── join_data.py                     # Data merging
    │
    └── crawler/
        ├── run_full_crawl.sh
        ├── scrapy.cfg
        └── imdbCrawler/
            └── spiders/
                └── top9000.py
```

---

## 📌 Quick Command Reference

```bash
# 1. Run data quality check
python src/data_quality_report.py

# 2. Run full pipeline (clean + features + model)
python main.py

# 3. Generate EDA report only
python src/eda_visualizations.py

# 4. Train and compare models
python src/model_training.py

# 5. Launch Gradio app
python src/gradio_app.py
```

---

## ✅ Success Criteria

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Dataset size | **19,000+ movies** ✅ (already have) | `len(df)` |
| Features | 40+ engineered | Feature list length |
| EDA plots | 25+ visualizations | Plot count |
| Models compared | 5+ algorithms | Comparison table |
| Best model R² | > 0.70 | Test set evaluation |
| Best model MAE | < $50M | Test set evaluation |
| Demo tabs | 6 functional | Gradio inspection |
| Code documentation | All modules | Docstrings |

---

## 🚀 Getting Started

Start with Phase 1 (Data Quality & Cleaning) and proceed sequentially. Each phase builds upon the previous one.

> **Note:** Your friend already crawled 19k movies. Load the full dataset from your local storage before starting.

```bash
# First, activate your environment
conda activate movie

# Copy your full 19k dataset to the project
# cp /path/to/full_data_joined.csv dataset/data_joined.csv

# Then start with data quality validation
python src/data_quality_report.py
```

**Good luck with your project! 🎬📊**

