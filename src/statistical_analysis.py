"""
Statistical Analysis Module
Performs comprehensive statistical tests and analysis on the dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for box office data"""
    
    def __init__(self, data_path='dataset/data_cleaned.csv'):
        self.data_path = data_path
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load cleaned data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} movies with {len(self.df.columns)} columns")
        return self
    
    def compute_summary_statistics(self):
        """Compute comprehensive summary statistics"""
        print("\n=== SUMMARY STATISTICS ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Key columns for detailed analysis
        key_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating', 'Rating_Count', 'ROI']
        available_key_cols = [col for col in key_cols if col in numeric_cols]
        
        if not available_key_cols:
            print("⚠ No numeric columns found for analysis")
            return self
        
        summary = pd.DataFrame()
        
        for col in available_key_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) > 0:
                summary[col] = {
                    'Count': len(col_data),
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Std': col_data.std(),
                    'Min': col_data.min(),
                    'Q1': col_data.quantile(0.25),
                    'Q3': col_data.quantile(0.75),
                    'Max': col_data.max(),
                    'Skewness': col_data.skew(),
                    'Kurtosis': col_data.kurt()
                }
        
        summary = summary.T
        self.results['summary_statistics'] = summary
        
        print(summary)
        
        # Save to CSV with proper index label
        summary.to_csv('demo/stats_summary.csv', index_label='Feature')
        print("\n✓ Summary statistics saved to demo/stats_summary.csv")
        
        return self
    
    def test_normality(self):
        """Test normality of key variables using Shapiro-Wilk test"""
        print("\n=== NORMALITY TESTS (Shapiro-Wilk) ===")
        
        test_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating']
        available_cols = [col for col in test_cols if col in self.df.columns]
        
        if not available_cols:
            return self
        
        normality_results = []
        
        for col in available_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) > 3 and len(col_data) < 5000:  # Shapiro-Wilk limits
                try:
                    stat, p_value = shapiro(col_data.sample(min(len(col_data), 5000)))
                    is_normal = p_value > 0.05
                    
                    normality_results.append({
                        'Column': col,
                        'Statistic': stat,
                        'P-Value': p_value,
                        'Is_Normal': 'Yes' if is_normal else 'No'
                    })
                    
                    print(f"{col}: p-value = {p_value:.6f} ({'Normal' if is_normal else 'Not Normal'})")
                except Exception as e:
                    print(f"{col}: Test failed - {e}")
        
        self.results['normality_tests'] = pd.DataFrame(normality_results)
        return self
    
    def compute_correlation_matrix(self):
        """Compute correlation matrix with p-values"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        numeric_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating', 'Rating_Count',
                       'Cast_Count', 'Crew_Count', 'Genre_Count', 'Release_Year']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("⚠ Not enough numeric columns for correlation analysis")
            return self
        
        # Compute correlation
        corr_data = self.df[available_cols].corr()
        
        # Compute p-values
        def calculate_pvalues(df):
            """Calculate p-values for correlations"""
            dfcols = pd.DataFrame(columns=df.columns)
            pvalues = dfcols.transpose().join(dfcols, how='outer')
            
            for r in df.columns:
                for c in df.columns:
                    if r == c:
                        pvalues[r][c] = 0
                    else:
                        try:
                            tmp = df[[r, c]].dropna()
                            if len(tmp) > 2:
                                _, p = stats.pearsonr(tmp[r], tmp[c])
                                pvalues[r][c] = p
                            else:
                                pvalues[r][c] = 1
                        except:
                            pvalues[r][c] = 1
            return pvalues
        
        pvalues = calculate_pvalues(self.df[available_cols])
        
        self.results['correlation_matrix'] = corr_data
        self.results['correlation_pvalues'] = pvalues
        
        # Save to CSV
        corr_data.to_csv('demo/correlation_matrix.csv')
        print("✓ Correlation matrix saved to demo/correlation_matrix.csv")
        
        # Find strongest correlations
        print("\n=== STRONGEST CORRELATIONS ===")
        corr_pairs = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                col1 = corr_data.columns[i]
                col2 = corr_data.columns[j]
                corr_val = corr_data.iloc[i, j]
                if abs(corr_val) > 0.3:  # Moderate correlation threshold
                    corr_pairs.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': corr_val
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            print(corr_df.to_string(index=False))
        
        return self
    
    def check_multicollinearity(self):
        """Check for multicollinearity using VIF"""
        print("\n=== MULTICOLLINEARITY CHECK (VIF) ===")
        
        numeric_cols = ['Budget', 'Runtime', 'Rating', 'Rating_Count',
                       'Cast_Count', 'Crew_Count', 'Genre_Count', 'Release_Year']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("⚠ Not enough columns for VIF analysis")
            return self
        
        # Prepare data - drop NaN
        vif_data = self.df[available_cols].dropna()
        
        if len(vif_data) < 10:
            print("⚠ Not enough data for VIF analysis")
            return self
        
        try:
            vif_results = []
            for i, col in enumerate(available_cols):
                vif = variance_inflation_factor(vif_data.values, i)
                vif_results.append({
                    'Feature': col,
                    'VIF': vif,
                    'Status': 'High' if vif > 10 else 'Moderate' if vif > 5 else 'Low'
                })
            
            vif_df = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
            print(vif_df.to_string(index=False))
            
            self.results['vif'] = vif_df
            
            print("\nInterpretation:")
            print("  VIF < 5: Low multicollinearity")
            print("  5 < VIF < 10: Moderate multicollinearity")
            print("  VIF > 10: High multicollinearity (consider removing)")
            
        except Exception as e:
            print(f"⚠ VIF calculation failed: {e}")
        
        return self
    
    def analyze_outliers(self):
        """Analyze outliers using IQR and Z-score methods"""
        print("\n=== OUTLIER ANALYSIS ===")
        
        outlier_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating_Count']
        available_cols = [col for col in outlier_cols if col in self.df.columns]
        
        outlier_results = []
        
        for col in available_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) < 10:
                continue
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = (z_scores > 3).sum()
            
            outlier_results.append({
                'Column': col,
                'IQR_Outliers': iqr_outliers,
                'IQR_Percent': f"{(iqr_outliers / len(col_data) * 100):.2f}%",
                'Z_Outliers': z_outliers,
                'Z_Percent': f"{(z_outliers / len(col_data) * 100):.2f}%"
            })
        
        outlier_df = pd.DataFrame(outlier_results)
        print(outlier_df.to_string(index=False))
        
        self.results['outliers'] = outlier_df
        return self
    
    def generate_insights(self):
        """Generate textual insights from statistical analysis"""
        print("\n=== KEY INSIGHTS ===")
        
        insights = []
        
        # Insight 1: Data size
        insights.append(f"📊 Dataset contains {len(self.df)} movies")
        
        # Insight 2: Gross distribution
        if 'Gross_worldwide' in self.df.columns:
            gross = self.df['Gross_worldwide'].dropna()
            median_gross = gross.median()
            mean_gross = gross.mean()
            insights.append(f"💰 Median gross: ${median_gross/1e6:.1f}M, Mean: ${mean_gross/1e6:.1f}M")
            
            if mean_gross > median_gross * 1.5:
                insights.append("   → Distribution is right-skewed (few blockbusters pull up the mean)")
        
        # Insight 3: Budget vs Gross correlation
        if 'correlation_matrix' in self.results:
            corr_matrix = self.results['correlation_matrix']
            if 'Budget' in corr_matrix.columns and 'Gross_worldwide' in corr_matrix.columns:
                corr_val = corr_matrix.loc['Budget', 'Gross_worldwide']
                insights.append(f"🎯 Budget-Gross correlation: {corr_val:.3f}")
                
                if corr_val > 0.7:
                    insights.append("   → Strong positive correlation: Higher budgets → higher gross")
                elif corr_val > 0.5:
                    insights.append("   → Moderate correlation: Budget matters but isn't everything")
                else:
                    insights.append("   → Weak correlation: Success depends on many factors")
        
        # Insight 4: Rating impact
        if 'Rating' in self.df.columns and 'Gross_worldwide' in self.df.columns:
            high_rated = self.df[self.df['Rating'] >= 8.0]['Gross_worldwide'].median()
            low_rated = self.df[self.df['Rating'] < 6.0]['Gross_worldwide'].median()
            
            if pd.notna(high_rated) and pd.notna(low_rated):
                insights.append(f"⭐ High-rated (≥8.0) median gross: ${high_rated/1e6:.1f}M")
                insights.append(f"   Low-rated (<6.0) median gross: ${low_rated/1e6:.1f}M")
        
        for insight in insights:
            print(insight)
        
        self.results['insights'] = insights
        return self
    
    def run_full_analysis(self):
        """Execute all statistical analyses"""
        print("\n" + "="*70)
        print(" STATISTICAL ANALYSIS")
        print("="*70)
        
        (self
            .load_data()
            .compute_summary_statistics()
            .test_normality()
            .compute_correlation_matrix()
            .check_multicollinearity()
            .analyze_outliers()
            .generate_insights())
        
        print("\n" + "="*70)
        print(" STATISTICAL ANALYSIS COMPLETED")
        print("="*70)
        
        return self
    
    def get_results(self):
        """Return all analysis results"""
        return self.results


if __name__ == "__main__":
    analyzer = StatisticalAnalyzer('dataset/data_cleaned.csv')
    analyzer.run_full_analysis()

