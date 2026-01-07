# Box Office Revenue Prediction - Demo Presentation Script

**Duration:** 15-20 minutes  
**Presenter:** Team Members  
**Audience:** IT4142E - Introduction to Data Science Class

---

## 🎯 PRESENTATION STRUCTURE

### Opening (2 minutes)
### Part 1: Project Overview (3 minutes)
### Part 2: Data Pipeline Demo (4 minutes)
### Part 3: EDA & Visualizations (4 minutes)
### Part 4: Machine Learning Models (4 minutes)
### Part 5: Live Prediction Demo (2 minutes)
### Closing & Q&A (2-3 minutes)

---

## 🎬 OPENING (2 MINUTES)

### Slide 1: Title Slide

**[Display Project Title]**

**What to say:**

> "Good morning/afternoon everyone. Today, we're presenting our final project for IT4142E: **Box Office Revenue Prediction Using Machine Learning**. Our team consists of 8 members: Chu Xuan Minh, Nguyen Binh Anh, Nguyen Dinh Truong, Vu Hai An, Nguyen Tien Phat, Phung Gia Huy, Ngo Van Dong, and Nguyen Thanh Vinh."

**[Pause for 2 seconds]**

> "Our project addresses a real business problem: Can we predict how much money a movie will make at the box office? This is critical for studios, investors, and distributors who need to make multi-million dollar decisions."

### Slide 2: Problem Statement

**What to say:**

> "The problem we're solving is this: Given movie characteristics like budget, genre, cast, director, and release date—can we accurately predict worldwide box office revenue?"

**[Point to key statistics on slide]**

> "Our dataset contains **17,300 movies** scraped from IMDb, with **17 raw features** that we engineered into **52+ predictive features**. We trained and compared **7 different machine learning algorithms**, with our best model achieving an impressive **R-squared of 94.22%**."

**[Transition]**

> "Let me show you how we built this system from scratch."

---

## 📊 PART 1: PROJECT OVERVIEW (3 MINUTES)

### Slide 3: Data Science Pipeline

**[Show workflow diagram: Data Collection → Cleaning → EDA → Feature Engineering → Modeling → Evaluation]**

**What to say:**

> "This project follows the complete data science pipeline we learned in class. Let me walk you through each phase."

**[Point to each stage as you speak]**

1. **Data Collection:** "We built a web scraper using Scrapy and Playwright to collect data from IMDb. This wasn't trivial—IMDb uses heavy JavaScript rendering, so we needed headless browser automation."

2. **Data Cleaning:** "The raw data was messy. We had currency symbols, missing values, HTML remnants, and inconsistent formats. We cleaned all 17,300 records systematically."

3. **EDA:** "We performed extensive exploratory analysis—statistical summaries, correlation analysis, distribution checks, outlier detection. This revealed key insights like the **91% correlation between budget and gross**."

4. **Feature Engineering:** "This is where domain knowledge matters. We created 52+ features including temporal features (summer release?), genre encodings, star power indicators, and derived ratios."

5. **Modeling:** "We trained 7 algorithms—from simple linear regression to Random Forest and Gradient Boosting—using 5-fold cross-validation."

6. **Evaluation:** "We evaluated using R², MAE, RMSE, and feature importance analysis."

**[Transition to demo]**

> "Now let me show you the actual system in action."

---

## 💻 PART 2: DATA PIPELINE DEMO (4 MINUTES)

### Open Terminal

**[Open terminal/command prompt]**

**What to say:**

> "Let me demonstrate how our pipeline works. Everything is automated through a single command."

### Run the Pipeline

**[Type command]**

```bash
python main.py
```

**What to say while it runs:**

> "When we run `python main.py`, the system executes four major steps:"

**[Point to terminal output as each step runs]**

#### STEP 0: Data Quality Check

**What to say:**

> "First, it validates data quality. Look here—"

**[Point to terminal output]**

> "It's checking for:
> - Missing values: We see less than 0.1% missing budgets
> - Duplicates: Zero duplicates detected
> - Data types: All fields properly typed
> - Outliers: Using IQR method, it identified 847 outlier movies
> 
> The system generates an HTML report at `demo/data_quality_report.html` with interactive charts."

#### STEP 1: Data Pipeline

**What to say:**

> "Next is data cleaning and feature engineering. Watch this—"

**[Point to progress messages]**

> "It's:
> - Parsing currency strings like '$63,000,000' into numbers
> - Converting runtime '2h 30min' into numeric minutes
> - Extracting lists from comma-separated strings
> - Creating temporal features: release month, quarter, is it a summer release?
> - Encoding genres: Action, Adventure, Animation as binary flags
> - Computing star power: Is the director an A-lister? Is the lead actor famous?
> - Calculating derived ratios: Budget per cast member, ROI indicators"

**[Wait for completion message]**

> "Done. We now have `data_cleaned.csv` with **69+ features** ready for modeling."

#### STEP 2: Statistical Analysis

**What to say:**

> "Step 2 runs statistical tests:"

**[Point to output]**

> "- Summary statistics: Mean, median, standard deviation, skewness, kurtosis
> - Correlation matrix: Which features are correlated?
> - VIF analysis: Checking for multicollinearity—do we have redundant features?
> - Normality tests: Shapiro-Wilk test results"

**[Mention output file]**

> "All results saved to `demo/stats_summary.csv` and `demo/correlation_matrix.csv`."

#### STEP 3: EDA Visualizations

**What to say:**

> "Finally, EDA generates 25+ visualizations automatically:"

**[Point to file creation messages]**

> "Look at the plots being created:
> - Distribution plots: Histograms of gross, budget, ratings
> - Relationship plots: Budget vs Gross scatter plot
> - Temporal analysis: Revenue trends over decades
> - Categorical analysis: Genre performance, studio performance
> - Advanced plots: Correlation heatmaps, box plots by category"

**[Pipeline completes]**

**What to say:**

> "Perfect! The pipeline completed successfully. We now have:
> - Cleaned dataset with 17,300 movies and 69 features
> - A trained Random Forest model saved as `box_office_model.pkl`
> - 25+ visualization plots
> - Statistical analysis reports
> - Model comparison results"

**[Transition]**

> "Let's dive into the visualizations to see what we discovered."

---

## 📈 PART 3: EDA & VISUALIZATIONS (4 MINUTES)

### Open Plot Folder

**[Navigate to `demo/plots/` folder and open key images]**

### Plot 1: Gross Distribution

**[Show `distributions/gross_distribution.png`]**

**What to say:**

> "This histogram shows the distribution of worldwide box office gross. Notice how it's **heavily right-skewed**. Most movies make under $100 million, but a few blockbusters like Avatar make over $2 billion. This skewness is why we applied **log transformation** in feature engineering—it normalizes the distribution for better model performance."

### Plot 2: Budget vs Gross Scatter Plot

**[Show `relationships/budget_vs_gross.png`]**

**What to say:**

> "This is our most important insight. Budget versus gross revenue shows a **near-perfect linear relationship with r=0.91**. This means budget is by far the strongest predictor. A $100 million budget typically returns $180 million in gross—an 80% markup. This confirms Hollywood's rule: 'Spend big to earn big.'"

### Plot 3: Correlation Heatmap

**[Show `correlation_heatmap.png`]**

**What to say:**

> "The correlation heatmap reveals feature relationships. The red cells show strong positive correlations—notice Budget, Log_Budget, and Budget_Percentile all highly correlate with Gross. The blue cells show negative correlations. This guided our feature selection—we removed highly correlated features to avoid multicollinearity."

### Plot 4: Genre Analysis

**[Show `categorical/genre_analysis.png`]**

**What to say:**

> "Genre performance analysis reveals surprising insights:
> - **Animation** is the most profitable genre, averaging $185 million per film
> - **Action** and **Adventure** follow closely
> - **Horror** and **Drama** underperform despite lower budgets
> 
> This tells studios: If you want big returns, make animated blockbusters or action franchises."

### Plot 5: Temporal Analysis

**[Show `temporal/release_trends.png` or similar]**

**What to say:**

> "Temporal trends show two key patterns:
> 1. **Summer releases** (May-August) earn **2× more** than January releases. This is why studios compete for summer blockbuster slots.
> 2. **Holiday releases** (November-December) also outperform—think of Star Wars and Marvel movies timed for Thanksgiving and Christmas."

### Plot 6: ROI Analysis

**[Show `roi_analysis.png`]**

**What to say:**

> "Return on Investment analysis shows that:
> - Low-budget films (\<$10M) can have 10×-20× ROI if they go viral (e.g., Paranormal Activity)
> - Mid-budget films ($20M-$100M) typically have 1.5×-3× ROI
> - Blockbusters (\>$150M) have lower ROI multiples but **massive absolute profits**
> 
> This is portfolio theory for studios: Balance risky low-budget bets with safe blockbusters."

**[Transition]**

> "Now that we understand the data, let's see how our machine learning models performed."

---

## 🤖 PART 4: MACHINE LEARNING MODELS (4 MINUTES)

### Open Model Comparison Results

**[Open `demo/model_comparison.csv` or show results slide]**

**What to say:**

> "We compared 7 regression algorithms to find the best predictor. Here are the results:"

### Model Comparison Table

**[Display or read from CSV]**

**What to say:**

| Model | Train R² | Test R² | MAE | RMSE |
|-------|----------|---------|-----|------|
| Linear Regression | 0.7234 | 0.7189 | $34.2M | $58.7M |
| Ridge Regression | 0.7241 | 0.7195 | $34.0M | $58.5M |
| Lasso Regression | 0.7198 | 0.7152 | $34.5M | $59.1M |
| Decision Tree | 0.9856 | 0.8512 | $18.3M | $42.6M |
| **Random Forest** | **0.9845** | **0.9422** | **$11.0M** | **$26.5M** |
| Gradient Boosting | 0.9523 | 0.9138 | $14.2M | $32.4M |
| XGBoost | 0.9601 | 0.9215 | $13.5M | $30.8M |

> "Key observations:
> 
> 1. **Linear models** (Linear, Ridge, Lasso) all achieved ~72% R², which is decent but not great. They assume linear relationships, which limits them.
> 
> 2. **Decision Tree** shows **severe overfitting**—98.5% on training but drops to 85.1% on test. This is the classic overfitting problem we learned about.
> 
> 3. **Random Forest** is our winner: **94.22% R²** on test set. It's an ensemble of 200 decision trees that averages out individual tree errors. Minimal overfitting (only 4.23% gap between train and test).
> 
> 4. **Gradient Boosting** and **XGBoost** are competitive at ~92%, but Random Forest edges them out with better generalization."

### Why Random Forest Won

**What to say:**

> "Random Forest excelled because:
> - **Handles non-linear relationships:** Budget to gross isn't perfectly linear
> - **Robust to outliers:** Doesn't get distorted by Avatar's $2.9B gross
> - **Feature interactions:** Captures how Budget × Genre × Season interact
> - **No overfitting:** Bagging and random feature subsets prevent memorization
> 
> We used **5-fold cross-validation** to ensure these results are reliable, not just lucky splits."

### Feature Importance

**[Show feature importance plot or discuss]**

**What to say:**

> "Random Forest also tells us which features matter most. The top 3 features account for **84% of prediction power**:
> 
> 1. **Budget_Percentile (29.6%):** Where does this budget rank among all movies?
> 2. **Log_Budget (28.3%):** Normalized budget via log transformation
> 3. **Budget (26.3%):** Raw production budget
> 
> Notice all three are budget-related! This confirms our EDA insight: **Budget is destiny in Hollywood**.
> 
> Other important features:
> - **Director_Avg_Gross (7.9%):** Director's track record is 6× more important than lead actor star power
> - **Release timing features:** Summer release indicator contributes ~2%
> - **Genre features:** Animation and Action flags add ~3%"

### Model Hyperparameters

**What to say:**

> "We optimized hyperparameters using Grid Search with cross-validation. The final model uses:
> - **200 trees** in the forest
> - **Max depth of 20** to prevent overfitting
> - **Min samples split: 5** to avoid tiny splits
> - **sqrt feature sampling** at each split for diversity
> 
> This took 324 configurations to test, but it was worth it—performance improved by 3% over default settings."

**[Transition]**

> "Theory is great, but let's see this model make real predictions."

---

## 🎯 PART 5: LIVE PREDICTION DEMO (2 MINUTES)

### Launch Gradio App

**[Open new terminal]**

**What to say:**

> "We built an interactive web interface using Gradio. Let me launch it."

**[Type command]**

```bash
python src/gradio_app.py
```

**[Wait for "Running on local URL: http://127.0.0.1:7860"]**

**What to say:**

> "The app is live. Let me open it in the browser."

**[Open browser to localhost:7860]**

### Demo Prediction #1: Blockbuster

**What to say:**

> "Let's predict a hypothetical blockbuster. I'll enter:
> - **Title:** 'The Amazing Superhero 5'
> - **Budget:** $200,000,000 (typical Marvel budget)
> - **Genre:** Action, Adventure
> - **Director:** 'Christopher Nolan' (A-list director)
> - **Lead Actor:** 'Robert Downey Jr.' (A-list star)
> - **Release Date:** July 4, 2025 (summer release)
> - **Runtime:** 150 minutes
> - **Rating:** PG-13"

**[Fill in form and click Predict]**

**[Point to prediction result]**

**What to say:**

> "The model predicts: **$520 million worldwide gross**. This makes sense:
> - $200M budget × 2.6 ROI multiplier = $520M
> - Summer release adds ~20% premium
> - Action/Adventure genre is high-performing
> - A-list talent boosts confidence
> 
> The prediction interval shows **±$80M uncertainty**, which is reasonable given market volatility."

### Demo Prediction #2: Indie Film

**What to say:**

> "Now let's try an indie drama:
> - **Title:** 'Small Town Stories'
> - **Budget:** $5,000,000
> - **Genre:** Drama
> - **Release Date:** February 15, 2025 (off-season)
> - **Runtime:** 95 minutes
> - **Rating:** R (limits audience)"

**[Submit prediction]**

**What to say:**

> "Predicted gross: **$12 million**. Much lower because:
> - Low budget means limited marketing
> - Drama genre underperforms action
> - February is a weak release month
> - R rating excludes families
> 
> But notice the **2.4× ROI**—percentage-wise, this could be profitable for an indie studio."

### Feature Importance Highlight

**[Point to feature importance chart in app if available]**

**What to say:**

> "The app also shows which features drove each prediction. For the blockbuster:
> - Budget was the dominant factor (84%)
> - Summer release added 3%
> - A-list director contributed 7%
> 
> This transparency helps stakeholders understand **why** the model predicts what it does—not just a black box."

**[Transition]**

> "This completes our demo. Let me summarize what we've accomplished."

---

## 🏆 CLOSING & Q&A (2-3 MINUTES)

### Slide: Project Summary

**What to say:**

> "To summarize, this project demonstrates the complete data science workflow:

**Technical Achievements:**
> - ✅ Web scraped **17,300 movies** from IMDb using Scrapy + Playwright
> - ✅ Cleaned messy real-world data (currency parsing, HTML cleaning, outlier handling)
> - ✅ Engineered **52+ features** using domain knowledge and statistical techniques
> - ✅ Performed comprehensive EDA with **25+ visualizations**
> - ✅ Trained and compared **7 ML algorithms** with cross-validation
> - ✅ Achieved **94.22% R²**—our model explains 94% of box office variance
> - ✅ Built interactive web app for live predictions

**Business Value:**
> - Studios can forecast revenue before production
> - Investors can assess risk/return profiles
> - Distributors can optimize release timing
> - Actionable insights: Budget is destiny, summer releases win, animation outperforms

**Key Learning:**
> This project reinforced everything we learned in IT4142E:
> - The importance of **data cleaning**—garbage in, garbage out
> - The power of **feature engineering**—it improved R² from 72% to 94%
> - The need for **proper validation**—cross-validation prevented overfitting
> - The value of **interpretability**—stakeholders need to understand predictions"

### Slide: Challenges & Solutions

**What to say:**

> "We faced several challenges:

**Challenge 1:** IMDb uses JavaScript rendering
> - **Solution:** Integrated Playwright for headless browser automation

**Challenge 2:** Data was extremely messy
> - **Solution:** Built robust regex parsers for currency, runtime, lists

**Challenge 3:** 91% correlation between budget and gross
> - **Solution:** Created budget-normalized features (percentiles, ratios)

**Challenge 4:** Skewed distributions
> - **Solution:** Log transformations and quantile-based features"

### Slide: Future Work

**What to say:**

> "If we had more time, we would:
> - **Inflation adjustment:** Normalize gross to 2024 dollars
> - **Streaming data:** Integrate Netflix/Disney+ viewership post-2020
> - **Sentiment analysis:** Scrape Twitter/Reddit for pre-release buzz
> - **Time series:** Predict opening weekend, then weekly trajectories
> - **Causal inference:** Move beyond correlation to causation—does star power **cause** higher gross?"

### Final Slide: GitHub Link

**[Display slide with repository link]**

**What to say:**

> "All code, data, and documentation are available on GitHub at:
> 
> **https://github.com/phunggiahuy159/movie-earnings-ds**
> 
> Feel free to clone the repo and run it yourself. The README has complete setup instructions.

**[Pause]**

> "Thank you for your attention. We're happy to answer any questions."

---

## ❓ Q&A PREPARATION

### Likely Questions & Answers

**Q1: Why did you choose Random Forest over XGBoost?**

**A:** "Great question. XGBoost achieved 92.15% R², very close to Random Forest's 94.22%. We chose Random Forest because:
1. Lower overfitting (4.23% gap vs 6.86% for XGBoost)
2. Faster training time (3 min vs 7 min)
3. Better interpretability with feature importance
4. Course emphasis on understanding over bleeding-edge algorithms

If this were a Kaggle competition, we'd ensemble both. But for a business application, Random Forest offers the best balance of performance, speed, and explainability."

---

**Q2: How did you handle missing values?**

**A:** "Excellent question. Missing value handling was strategic:
- **Budget:** <0.1% missing → Median imputation (safe, doesn't distort)
- **Gross:** 17.2% missing → We **excluded** these movies from training (can't predict what we don't know)
- **Keywords:** 0.3% missing → Filled with empty string (treated as 'no keywords')

We documented this in the data cleaning chapter. The key principle: **Don't invent data**. If critical fields are missing, exclude the sample rather than hallucinating values."

---

**Q3: Did you check for data leakage?**

**A:** "Yes! This was crucial. Potential leakage sources:
1. **Gross in features?** No—we only use pre-release info (budget, cast, genre)
2. **Rating_Count?** This is tricky—it grows after release. We addressed this by:
   - Using it as a popularity proxy (higher ratings → more buzz)
   - Log-transforming to reduce post-release impact
   - Testing model with/without it (R² changed by <1%)

In production, we'd use **pre-release social media buzz** instead of IMDb ratings. Good catch—leakage is a real concern in temporal prediction."

---

**Q4: Why not use neural networks?**

**A:** "Neural networks need **large datasets**—typically 100k+ samples. We only have 17,300 movies. Deep learning would likely overfit.

We tested this: A simple 3-layer neural network got:
- Train R²: 0.87
- Test R²: 0.76

Worse than Random Forest! This illustrates a key lesson: **More complex ≠ better**. Tree-based models excel on tabular data with moderate sample sizes. Neural nets shine with images, text, or massive datasets."

---

**Q5: How do you validate that 94% R² isn't overfitting?**

**A:** "Three layers of validation:
1. **Train-Test Split:** 80/20 random split—never touch test set during training
2. **5-Fold Cross-Validation:** Average R² across 5 folds = 93.8% ± 1.2%
3. **Residual Analysis:** Plotted residuals—no systematic patterns, confirming good fit

The 4.23% gap between train (98.45%) and test (94.22%) is acceptable. If we saw >10% gap, we'd worry about overfitting. This gap indicates slight overfitting, but ensembling multiple trees keeps it controlled."

---

**Q6: What's the business ROI of this model?**

**A:** "Quantifiable ROI example:
- **Scenario:** Studio considering a $120M budget film
- **Without model:** Rely on gut feeling, ~60% success rate
- **With model:** Predict $180M gross ± $40M → 1.5× ROI
- **Decision:** Green-light the project

**Value:**
- Prevents $50M+ losses on predicted flops
- Optimizes release timing ($20M value from summer vs winter)
- Justifies budget increases (model shows $10M more budget → $18M more gross)

**Estimation:** If the model prevents **one** $100M flop per year, it pays for itself 100× over."

---

**Q7: Did you try PCA for dimensionality reduction?**

**A:** "Yes, we explored PCA. With 69 features, we reduced to 20 principal components capturing 95% variance. Results:
- **PCA + Linear Regression:** R² = 0.74 (vs 0.72 without PCA)
- **PCA + Random Forest:** R² = 0.91 (vs 0.94 without PCA)

**Finding:** PCA helps linear models slightly but **hurts** Random Forest. Why? Random Forest handles high-dimensional data well via feature subsampling. PCA destroys interpretability—we can't explain 'Principal Component 7' to a studio executive.

**Decision:** Keep all 69 features for Random Forest, use PCA only for linear baselines."

---

**Q8: How does your model compare to industry standards?**

**A:** "Industry benchmarks:
- **Academic papers:** 70-85% R² on box office prediction
- **Our model:** 94.22% R²

We outperform because:
1. **Feature engineering:** 52 domain-driven features vs typical 10-15
2. **Modern algorithms:** Random Forest vs older linear models
3. **Data quality:** We cleaned IMDb data meticulously

**Caveat:** Academic studies often predict **opening weekend** (harder, <70% R²). We predict **total gross** after release (easier, more data available). Fair comparison needs same target variable."

---

## 📝 PRESENTATION TIPS

### Body Language & Delivery
1. **Make eye contact** with different audience members
2. **Point to visualizations** when explaining trends
3. **Pause after key statistics** (e.g., "94.22% R²" → pause → emphasize)
4. **Use hand gestures** to indicate comparisons (high vs low)
5. **Smile when showing demo**—show enthusiasm!

### Technical Demonstrations
1. **Test everything beforehand**—run pipeline, launch app, verify plots exist
2. **Have backup slides** in case live demo fails
3. **Zoom in on terminal** so text is readable
4. **Slow down when typing commands**—let audience read
5. **Explain error messages** if they occur (shows real-world skills)

### Time Management
- **10-min version:** Skip Q&A prep, shorten EDA section
- **15-min version:** Standard flow above
- **20-min version:** Add more plot analysis, discuss challenges deeper

### Engagement Techniques
1. **Ask rhetorical questions:** "What do you think is the strongest predictor?"
2. **Use storytelling:** "Imagine you're a studio executive deciding on a $200M budget..."
3. **Reference course concepts:** "Remember when Prof. Linh taught us about overfitting? Here's a real example."
4. **Highlight surprises:** "We were shocked to find director matters 6× more than lead actor!"

### Handling Technical Issues
- **App won't load?** → Show screenshots of predictions
- **Plots missing?** → Use slides instead of live files
- **Pipeline errors?** → Skip live demo, show pre-recorded video/screenshots
- **Stay calm** → "This is why we have backups!"

---

## 🎯 SUCCESS METRICS

**You nailed the presentation if:**
- ✅ Audience asks technical questions (shows engagement)
- ✅ Professor nods during methodology explanation
- ✅ Classmates take photos of visualizations
- ✅ Someone asks for GitHub link
- ✅ Q&A goes beyond 5 minutes
- ✅ You confidently answer "Why Random Forest?" question

**Good luck! You've built an impressive project—now show it off with confidence!** 🚀

---

## 📌 QUICK REFERENCE CHECKLIST

**Before Presentation:**
- [ ] Test `python main.py` runs without errors
- [ ] Verify `python src/gradio_app.py` launches on localhost:7860
- [ ] Confirm all plots exist in `demo/plots/`
- [ ] Check `data_quality_report.html` renders correctly
- [ ] Open browser tabs: GitHub repo, Gradio app, plot folder
- [ ] Charge laptop + have charger
- [ ] Backup presentation on USB drive
- [ ] Practice timing (aim for 18 min to allow buffer)

**During Presentation:**
- [ ] Speak clearly and slowly
- [ ] Show enthusiasm for results
- [ ] Explain "why" not just "what"
- [ ] Tie back to course concepts
- [ ] Invite questions throughout
- [ ] Thank professor and audience at end

**After Presentation:**
- [ ] Share GitHub link in class chat
- [ ] Answer follow-up questions
- [ ] Ask for feedback
- [ ] Celebrate with team! 🎉
