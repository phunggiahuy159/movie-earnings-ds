# 🎬 Box Office Prediction - Project Todo List

**Last Updated:** 2026-01-06  
**Goals:** Fix demo bugs, upgrade UI/UX, create LaTeX report

---

## Part 1: Demo UI/UX Upgrade & Bug Fixes

### 1.1 Critical Bug Fixes

- [x] **Fix Tab 2 (Data Quality) - iframe embedding broken**
  - **Issue:** Using `<iframe srcdoc="...">` causes content to not render properly
  - **Solution:** Replace with direct HTML rendering using `gr.HTML()` without iframe, or convert content to Markdown
  - **File:** `src/gradio_app.py` line 76-79

- [x] **Fix Tab 3 (EDA) - Looking for wrong subdirectory**
  - **Issue:** Code looks for `time_series/` but folder is named `temporal/`
  - **Solution:** Change `'time_series'` to `'temporal'` in `eda_tab()` method
  - **File:** `src/gradio_app.py` line 89

- [x] **Fix Tab 3 (EDA) - Missing `categorical/` subdirectory images**
  - **Issue:** Categorical folder may be empty or have different structure
  - **Solution:** Verify folder exists, add fallback handling

- [x] **Fix Static Plots - Not interactive**
  - **Issue:** All plots use static matplotlib, not interactive
  - **Cause:** `gr.Plot()` with matplotlib returns static image
  - **Solution:** Convert to Plotly for interactive charts (zoom, hover, pan)

### 1.2 UI/UX Enhancements

#### Visual Design Improvements
- [ ] **Upgrade theme for modern look**
  - Use custom CSS for glassmorphism effects
  - Add gradient backgrounds
  - Improve color palette consistency

- [ ] **Add loading indicators**
  - Show spinner when loading plots/data
  - Progress indication for predictions

- [ ] **Responsive layout improvements**
  - Better mobile/tablet support
  - Collapsible sections for long content

#### Tab 1 (Overview) Improvements
- [ ] Add animated hero section with key metrics
- [ ] Include sample movie posters/thumbnails
- [ ] Add quick navigation cards to other tabs

#### Tab 2 (Data Quality) Improvements
- [ ] Convert HTML report to native Gradio components
- [ ] Add interactive data quality metrics cards
- [ ] Show before/after cleaning comparison

#### Tab 3 (EDA) Improvements - HIGH PRIORITY
- [x] **Replace matplotlib with Plotly for interactivity**
  - Implement zoom, pan, hover tooltips
  - Add crosshair for precise value reading
- [ ] Add dropdown to select different plot types
- [ ] Create tabbed sub-navigation within EDA tab
- [ ] Add plot export functionality (PNG, SVG)

#### Tab 4 (Statistics) Improvements
- [x] Visualize correlation matrix as interactive heatmap
- [ ] Add hypothesis testing results display
- [ ] Color-code VIF values (green/yellow/red)

#### Tab 5 (Prediction) Improvements
- [x] Add genre selection dropdown
- [ ] Add release date picker
- [ ] Show similar movies from dataset
- [ ] Add confidence interval for predictions
- [ ] Animated result reveal

#### Tab 6 (Models) Improvements
- [x] Interactive model comparison chart with Plotly
- [ ] Add learning curves visualization
- [ ] Show cross-validation fold results
- [ ] Add model selection dropdown for detailed view

#### Tab 7 (Features) Improvements
- [x] Interactive feature importance with Plotly
- [ ] Add feature correlation network graph
- [ ] Show feature distributions on hover

### 1.3 Technical Debt

- [ ] Add proper error handling for missing files
- [ ] Implement lazy loading for plots
- [ ] Add caching for frequently accessed data
- [ ] Create configuration file for demo settings
- [ ] Add logging for debugging

---

## Part 2: LaTeX Report Construction

### 2.1 Report Structure (Following Course Requirements)

#### Front Matter
- [ ] **Title page**
  - Course: IT4142E - Introduction to Data Science
  - Project: Box Office Revenue Prediction
  - Team members
  - Date

- [ ] **Abstract** (200-300 words)
  - Problem statement
  - Methodology summary
  - Key results

- [ ] **Table of contents**

#### Chapter 1: Introduction (1-2 pages)
- [ ] **1.1 Problem Background**
  - Importance of box office prediction
  - Industry applications
  
- [ ] **1.2 Project Objectives**
  - Prediction goal (worldwide gross)
  - Descriptive analysis goals
  
- [ ] **1.3 Scope & Limitations**
  - Data coverage (IMDb, ~19,000 movies)
  - Time period
  - Feature limitations

#### Chapter 2: Data Collection (3-4 pages) - HIGH PRIORITY FOR TEACHER
- [ ] **2.1 Data Source Selection**
  - Why IMDb chosen over alternatives
  - API vs Scraping decision
  
- [ ] **2.2 Web Scraping Implementation**
  - Scrapy framework overview
  - **Include Scrapy spider code snippet** (top9000.py)
  - Playwright integration for JavaScript
  
- [ ] **2.3 Crawling Architecture**
  - URL frontier design
  - Pagination handling ("50 more" button)
  - robots.txt compliance
  
- [ ] **2.4 Data Fields Extracted** (17 fields)
  - Table with field names, types, examples
  
- [ ] **2.5 Challenges & Solutions**
  - JavaScript rendering issue → Playwright
  - Rate limiting → AUTOTHROTTLE
  - Dynamic content → DOM parsing after click

#### Chapter 3: Data Cleaning & Integration (3-4 pages) - HIGH PRIORITY FOR TEACHER
- [ ] **3.1 Raw Data Assessment**
  - **Include "Dirty Data" screenshots/examples**
  - Missing value counts
  - Data type issues
  
- [ ] **3.2 Missing Value Handling**
  - Strategy: Imputation by genre median
  - **Before/after comparison chart**
  - Code snippet from pipeline.py
  
- [ ] **3.3 Outlier Detection & Treatment**
  - IQR method explanation
  - **Box plot visualization showing outliers**
  - Justification for outlier handling
  
- [ ] **3.4 Data Standardization**
  - Studio name normalization (e.g., "Warner Bros" variants)
  - Genre standardization
  - Currency parsing ($XXX,XXX → int)
  
- [ ] **3.5 Data Validation**
  - Release date validation
  - Value range checks (rating 0-10, positive budget)
  
- [ ] **3.6 Integration Notes**
  - Single source (IMDb), but mention potential for multi-source integration

#### Chapter 4: Exploratory Data Analysis (4-5 pages) - HIGH PRIORITY FOR TEACHER
- [ ] **4.1 Summary Statistics**
  - Central tendency (Mean, Median, Mode)
  - Dispersion (Variance, Std Dev, IQR)
  - Shape (Skewness, Kurtosis)
  - **Table with all numeric features**
  
- [ ] **4.2 Univariate Analysis**
  - **Gross distribution (histogram + KDE)**
  - **Budget distribution (histogram + KDE)**
  - Q-Q plots for normality assessment
  - Log transformations applied
  
- [ ] **4.3 Bivariate Analysis**
  - **Budget vs Gross scatter plot with regression line**
  - Rating vs Gross analysis
  - Runtime vs Gross analysis
  
- [ ] **4.4 Multivariate Analysis**
  - **Correlation heatmap** (feature_importance.png)
  - Pairwise scatter matrix
  - PCA visualization (if applicable)
  
- [ ] **4.5 Categorical Analysis**
  - Genre performance comparison (bar chart)
  - Studio performance comparison
  - MPAA rating impact
  
- [ ] **4.6 Temporal Analysis**
  - Release year trends
  - Seasonal patterns (monthly/quarterly)
  - Decade-wise comparison
  
- [ ] **4.7 Key Insights from EDA**
  - "Budget has X correlation with Gross"
  - "Summer releases perform Y% better"
  - "Action genre has highest average gross"

#### Chapter 5: Feature Engineering (2-3 pages)
- [ ] **5.1 Temporal Features**
  - Release_Month, Quarter, Is_Summer, Is_Holiday, etc.
  
- [ ] **5.2 Budget Features**
  - Log_Budget, Budget_Tier, Budget_Percentile
  
- [ ] **5.3 Content Features**
  - Genre binaries, Is_Franchise, Is_Sequel
  
- [ ] **5.4 Star Power Features**
  - Has_A_List_Actor, Has_A_List_Director
  - A-list detection methodology
  
- [ ] **5.5 Studio Features**
  - Major studio detection
  - Studio track record
  
- [ ] **5.6 Feature Selection**
  - Correlation-based removal
  - VIF multicollinearity check
  - **Final feature count: 58**

#### Chapter 6: Machine Learning (3-4 pages)
- [ ] **6.1 Problem Formulation**
  - Regression task
  - Target variable: Gross_worldwide
  
- [ ] **6.2 Train/Test Split**
  - 80/20 split
  - Stratification considerations
  
- [ ] **6.3 Models Implemented**
  - Linear Regression (baseline)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - ElasticNet (L1+L2)
  - Decision Tree
  - **Random Forest (best performer)**
  - Gradient Boosting
  
- [ ] **6.4 Hyperparameter Tuning**
  - Grid search / cross-validation
  - Best parameters for each model
  
- [ ] **6.5 Cross-Validation**
  - **K-fold CV implementation** (K=5)
  - Results table

#### Chapter 7: Evaluation (2-3 pages)
- [ ] **7.1 Evaluation Metrics**
  - R² Score explanation
  - MAE explanation
  - RMSE explanation
  - MAPE explanation
  
- [ ] **7.2 Model Comparison Results**
  - **Table: All 7 models with Train/Test metrics**
  - Overfitting analysis (Train R² - Test R²)
  - **Bar chart comparing models**
  
- [ ] **7.3 Best Model Analysis**
  - Random Forest selected (Test R² = 0.94)
  - Feature importance ranking
  - **Feature importance bar chart**
  
- [ ] **7.4 Error Analysis**
  - Residual plots
  - Prediction vs Actual scatter
  - Where model fails (edge cases)

#### Chapter 8: Conclusion (1 page)
- [ ] **8.1 Summary of Findings**
  - Most important predictors
  - Model performance achieved
  
- [ ] **8.2 Limitations**
  - Data coverage limitations
  - Missing features (marketing budget, competition)
  
- [ ] **8.3 Future Work**
  - Deep learning approaches
  - More data sources
  - Real-time prediction API

#### Appendices
- [ ] **Appendix A: Code Structure**
  - Project directory tree
  - Module descriptions
  
- [ ] **Appendix B: Dataset Sample**
  - 10 sample rows from cleaned data
  
- [ ] **Appendix C: All EDA Plots**
  - Full 25+ visualization gallery
  
- [ ] **Appendix D: Complete Statistical Results**
  - Full VIF table
  - Complete correlation matrix

#### References
- [ ] Research papers on box office prediction
- [ ] Scrapy documentation
- [ ] scikit-learn documentation
- [ ] Course materials

### 2.2 Figures to Generate/Export

- [ ] **Fig 1:** Data pipeline flowchart
- [ ] **Fig 2:** Scrapy architecture diagram
- [ ] **Fig 3:** Raw vs cleaned data comparison
- [ ] **Fig 4:** Missing value heatmap
- [ ] **Fig 5:** Outlier box plots (budget, gross)
- [ ] **Fig 6:** Gross distribution with KDE
- [ ] **Fig 7:** Budget vs Gross scatter
- [ ] **Fig 8:** Correlation heatmap
- [ ] **Fig 9:** Genre performance bar chart
- [ ] **Fig 10:** Time trends line chart
- [ ] **Fig 11:** Model comparison bar chart
- [ ] **Fig 12:** Feature importance bar chart
- [ ] **Fig 13:** Actual vs Predicted scatter
- [ ] **Fig 14:** Residual plot

### 2.3 Tables to Create

- [ ] **Table 1:** Dataset fields (17 columns)
- [ ] **Table 2:** Missing value summary
- [ ] **Table 3:** Summary statistics (all features)
- [ ] **Table 4:** Feature engineering summary (52 features)
- [ ] **Table 5:** Model comparison results (7 models)
- [ ] **Table 6:** Top 20 feature importance
- [ ] **Table 7:** Cross-validation results

### 2.4 Code Snippets to Include

- [ ] Scrapy spider parse method
- [ ] Data cleaning parse_money() function
- [ ] Feature engineering example (star_power)
- [ ] Model training code
- [ ] Evaluation metrics calculation

---

## Priority Order

### Immediate (Today)
1. [x] Fix Tab 2 iframe bug
2. [x] Fix Tab 3 time_series → temporal path
3. [ ] Start LaTeX document structure

### This Week
4. [x] Convert matplotlib to Plotly (Tab 3, 5, 6, 7)
5. [ ] Complete Chapters 1-3 (Introduction, Data Collection, Cleaning)
6. [ ] Generate all figures for report

### Next Week
7. [ ] Complete Chapters 4-7 (EDA, Features, ML, Evaluation)
8. [ ] UI polish and theme improvements
9. [ ] Final review and submission

---

## Notes

### Teacher Preferences (from course_context.md)
- Focus 70% on Data Collection, Cleaning, EDA
- Show "dirty data" transformations
- Use Box Plots to justify outlier removal
- Include correlation heatmap before modeling
- Use Scrapy (not just CSV download)
- Avoid black box models - explain feature importance

### Technical Stack
- **Demo:** Gradio + Plotly (interactive)
- **Report:** LaTeX (Overleaf or local)
- **Figures:** Export from Plotly as PNG/PDF
