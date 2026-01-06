"""
Exploratory Data Analysis and Visualizations
Generate plots for understanding the box office dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAVisualizer:
    def __init__(self, data_path='dataset/data_cleaned.csv'):
        self.data_path = data_path
        self.df = None
        self.figures = {}
        
    def load_data(self):
        """Load cleaned data"""
        print("Loading cleaned data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} movies")
        return self
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        print("\n=== DATASET OVERVIEW ===")
        print(f"Total movies: {len(self.df)}")
        print(f"\nNumeric Features Summary:")
        
        numeric_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating', 'Rating_Count', 'ROI']
        summary = self.df[numeric_cols].describe()
        print(summary)
        
        return summary
    
    def plot_gross_distribution(self):
        """Plot distribution of worldwide gross"""
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        data = self.df['Gross_worldwide'].dropna()
        ax[0].hist(data / 1e6, bins=20, color='skyblue', edgecolor='black')
        ax[0].set_xlabel('Gross Worldwide (Millions $)')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Distribution of Worldwide Gross')
        ax[0].grid(True, alpha=0.3)
        
        # Box plot
        ax[1].boxplot(data / 1e6, vert=True)
        ax[1].set_ylabel('Gross Worldwide (Millions $)')
        ax[1].set_title('Box Plot of Worldwide Gross')
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures['gross_distribution'] = fig
        return fig
    
    def plot_budget_vs_gross(self):
        """Plot budget vs gross relationship"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = self.df.dropna(subset=['Budget', 'Gross_worldwide'])
        
        ax.scatter(data['Budget'] / 1e6, data['Gross_worldwide'] / 1e6, 
                   alpha=0.6, s=100, c='coral', edgecolors='black')
        
        # Add trend line
        z = np.polyfit(data['Budget'], data['Gross_worldwide'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['Budget'].min(), data['Budget'].max(), 100)
        ax.plot(x_line / 1e6, p(x_line) / 1e6, "r--", linewidth=2, label='Trend')
        
        ax.set_xlabel('Budget (Millions $)')
        ax.set_ylabel('Gross Worldwide (Millions $)')
        ax.set_title('Budget vs Worldwide Gross')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures['budget_vs_gross'] = fig
        return fig
    
    def plot_genre_analysis(self):
        """Analyze performance by genre"""
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count by genre
        genre_counts = self.df['Primary_Genre'].value_counts().head(10)
        ax[0].barh(genre_counts.index, genre_counts.values, color='steelblue')
        ax[0].set_xlabel('Count')
        ax[0].set_title('Top 10 Genres by Count')
        ax[0].grid(True, alpha=0.3, axis='x')
        
        # Average gross by genre
        genre_gross = self.df.groupby('Primary_Genre')['Gross_worldwide'].mean().sort_values(ascending=False).head(10)
        ax[1].barh(genre_gross.index, genre_gross.values / 1e6, color='lightcoral')
        ax[1].set_xlabel('Average Gross (Millions $)')
        ax[1].set_title('Top 10 Genres by Average Gross')
        ax[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self.figures['genre_analysis'] = fig
        return fig
    
    def plot_rating_analysis(self):
        """Analyze ratings vs gross"""
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Rating vs Gross
        data = self.df.dropna(subset=['Rating', 'Gross_worldwide'])
        ax[0].scatter(data['Rating'], data['Gross_worldwide'] / 1e6, 
                      alpha=0.6, s=100, c='mediumpurple', edgecolors='black')
        ax[0].set_xlabel('IMDb Rating')
        ax[0].set_ylabel('Gross Worldwide (Millions $)')
        ax[0].set_title('Rating vs Worldwide Gross')
        ax[0].grid(True, alpha=0.3)
        
        # Rating distribution
        ax[1].hist(self.df['Rating'].dropna(), bins=20, color='lightgreen', edgecolor='black')
        ax[1].set_xlabel('IMDb Rating')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Distribution of IMDb Ratings')
        ax[1].axvline(self.df['Rating'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {self.df["Rating"].mean():.2f}')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures['rating_analysis'] = fig
        return fig
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of numeric features"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        numeric_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating', 
                        'Rating_Count', 'Cast_Count', 'Genre_Count']
        
        corr_data = self.df[numeric_cols].dropna()
        correlation = corr_data.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap of Features')
        
        plt.tight_layout()
        self.figures['correlation_heatmap'] = fig
        return fig
    
    def plot_roi_analysis(self):
        """Analyze Return on Investment"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        roi_data = self.df.dropna(subset=['ROI']).copy()
        roi_data = roi_data[roi_data['ROI'] < 1000]  # Remove outliers
        
        roi_data_sorted = roi_data.sort_values('ROI', ascending=False).head(15)
        
        colors = ['green' if x > 0 else 'red' for x in roi_data_sorted['ROI']]
        ax.barh(range(len(roi_data_sorted)), roi_data_sorted['ROI'], color=colors)
        ax.set_yticks(range(len(roi_data_sorted)))
        ax.set_yticklabels(roi_data_sorted['Movie_Title'], fontsize=9)
        ax.set_xlabel('ROI (%)')
        ax.set_title('Top 15 Movies by Return on Investment')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        self.figures['roi_analysis'] = fig
        return fig
    
    # ==================== ADVANCED VISUALIZATIONS ====================
    
    def plot_distribution_with_kde(self, column, log_scale=False):
        """Histogram with kernel density estimate overlay"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = self.df[column].dropna()
        if log_scale:
            data = np.log10(data.replace(0, np.nan)).dropna()
        
        ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        data.plot(kind='kde', ax=ax, color='red', linewidth=2, label='KDE')
        
        ax.set_xlabel(f"{column} {'(log scale)' if log_scale else ''}")
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {column} with KDE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures[f'dist_kde_{column}'] = fig
        return fig
    
    def plot_qq(self, column):
        """Q-Q plot for normality assessment"""
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        data = self.df[column].dropna()
        stats.probplot(data, dist="norm", plot=ax)
        
        ax.set_title(f'Q-Q Plot: {column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures[f'qq_{column}'] = fig
        return fig
    
    def plot_pairwise_scatter(self, columns):
        """Scatter matrix for multiple variables"""
        import pandas as pd
        
        data = self.df[columns].dropna()
        
        fig = pd.plotting.scatter_matrix(data, figsize=(12, 12), alpha=0.5, 
                                          diagonal='kde', hist_kwds={'bins': 20})
        plt.suptitle('Pairwise Scatter Matrix', y=1.0)
        plt.tight_layout()
        
        self.figures['pairwise_scatter'] = fig[0][0].get_figure()
        return self.figures['pairwise_scatter']
    
    def plot_categorical_comparison(self, cat_col, num_col):
        """Violin/box plot comparing categories"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        data = self.df[[cat_col, num_col]].dropna()
        
        # Violin plot
        categories = data[cat_col].value_counts().head(8).index
        plot_data = data[data[cat_col].isin(categories)]
        
        sns.violinplot(data=plot_data, x=cat_col, y=num_col, ax=axes[0])
        axes[0].set_title(f'{num_col} by {cat_col} (Violin Plot)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot with individual points
        sns.boxplot(data=plot_data, x=cat_col, y=num_col, ax=axes[1])
        axes[1].set_title(f'{num_col} by {cat_col} (Box Plot)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures[f'categorical_{cat_col}_{num_col}'] = fig
        return fig
    
    def plot_time_trends(self):
        """Multi-line plot of trends over years"""
        if 'Release_Year' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trend 1: Average Budget over time
        if 'Budget' in self.df.columns:
            budget_trend = self.df.groupby('Release_Year')['Budget'].mean() / 1e6
            axes[0, 0].plot(budget_trend.index, budget_trend.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Average Budget Over Time')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Budget (Millions $)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Trend 2: Average Gross over time
        if 'Gross_worldwide' in self.df.columns:
            gross_trend = self.df.groupby('Release_Year')['Gross_worldwide'].mean() / 1e6
            axes[0, 1].plot(gross_trend.index, gross_trend.values, marker='o', 
                           color='green', linewidth=2)
            axes[0, 1].set_title('Average Gross Over Time')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Gross (Millions $)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Trend 3: Average Rating over time
        if 'Rating' in self.df.columns:
            rating_trend = self.df.groupby('Release_Year')['Rating'].mean()
            axes[1, 0].plot(rating_trend.index, rating_trend.values, marker='o', 
                           color='orange', linewidth=2)
            axes[1, 0].set_title('Average Rating Over Time')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('IMDb Rating')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Trend 4: Number of releases per year
        release_counts = self.df['Release_Year'].value_counts().sort_index()
        axes[1, 1].bar(release_counts.index, release_counts.values, color='purple', alpha=0.7)
        axes[1, 1].set_title('Number of Releases Per Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['time_trends'] = fig
        return fig
    
    def plot_seasonal_patterns(self):
        """Monthly/quarterly performance patterns"""
        if 'Release_Month' not in self.df.columns or 'Gross_worldwide' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Monthly patterns - filter out NaN months
        monthly_gross = self.df.groupby('Release_Month')['Gross_worldwide'].agg(['mean', 'median']) / 1e6
        # Remove NaN index if present
        monthly_gross = monthly_gross[monthly_gross.index.notna()]
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        x = np.arange(len(monthly_gross))
        width = 0.35
        
        axes[0].bar(x - width/2, monthly_gross['mean'], width, label='Mean', alpha=0.8)
        axes[0].bar(x + width/2, monthly_gross['median'], width, label='Median', alpha=0.8)
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Gross (Millions $)')
        axes[0].set_title('Average Gross by Release Month')
        axes[0].set_xticks(x)
        # Convert month indices to int and get month names
        axes[0].set_xticklabels([month_names[int(i)-1] for i in monthly_gross.index], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Quarterly patterns
        if 'Release_Quarter' in self.df.columns:
            quarterly_gross = self.df.groupby('Release_Quarter')['Gross_worldwide'].mean() / 1e6
            # Remove NaN index if present
            quarterly_gross = quarterly_gross[quarterly_gross.index.notna()]
            
            # Dynamically assign colors based on available quarters
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            quarter_colors = [colors[int(q)-1] for q in quarterly_gross.index]
            
            axes[1].bar(quarterly_gross.index, quarterly_gross.values, color=quarter_colors)
            axes[1].set_xlabel('Quarter')
            axes[1].set_ylabel('Average Gross (Millions $)')
            axes[1].set_title('Average Gross by Quarter')
            axes[1].set_xticks(quarterly_gross.index)
            axes[1].set_xticklabels([f'Q{int(q)}' for q in quarterly_gross.index])
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['seasonal_patterns'] = fig
        return fig
    
    def plot_budget_tier_analysis(self):
        """Analysis by budget tiers"""
        if 'Budget_Tier' not in self.df.columns or 'Gross_worldwide' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count by tier
        tier_counts = self.df['Budget_Tier'].value_counts()
        axes[0].bar(tier_counts.index, tier_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Budget Tier')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Number of Movies by Budget Tier')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Average gross by tier
        tier_gross = self.df.groupby('Budget_Tier')['Gross_worldwide'].mean() / 1e6
        tier_order = ['Micro', 'Low', 'Medium', 'High', 'Blockbuster']
        tier_gross = tier_gross.reindex([t for t in tier_order if t in tier_gross.index])
        
        axes[1].bar(range(len(tier_gross)), tier_gross.values, 
                   color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db'])
        axes[1].set_xlabel('Budget Tier')
        axes[1].set_ylabel('Average Gross (Millions $)')
        axes[1].set_title('Average Gross by Budget Tier')
        axes[1].set_xticks(range(len(tier_gross)))
        axes[1].set_xticklabels(tier_gross.index, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['budget_tier_analysis'] = fig
        return fig
    
    def plot_star_power_impact(self):
        """Analyze impact of A-list talent"""
        if 'Has_A_List_Actor' not in self.df.columns or 'Gross_worldwide' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # A-list actor impact
        actor_comparison = self.df.groupby('Has_A_List_Actor')['Gross_worldwide'].agg(['mean', 'median']) / 1e6
        
        x = np.arange(len(actor_comparison))
        width = 0.35
        
        axes[0].bar(x - width/2, actor_comparison['mean'], width, label='Mean', alpha=0.8)
        axes[0].bar(x + width/2, actor_comparison['median'], width, label='Median', alpha=0.8)
        axes[0].set_xlabel('Has A-List Actor')
        axes[0].set_ylabel('Gross (Millions $)')
        axes[0].set_title('Box Office Performance: With vs Without A-List Actors')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['No A-List', 'Has A-List'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Top actor count impact
        if 'Top_Actor_Count' in self.df.columns:
            actor_count_gross = self.df.groupby('Top_Actor_Count')['Gross_worldwide'].mean() / 1e6
            axes[1].bar(actor_count_gross.index, actor_count_gross.values, color='gold', alpha=0.7)
            axes[1].set_xlabel('Number of A-List Actors')
            axes[1].set_ylabel('Average Gross (Millions $)')
            axes[1].set_title('Average Gross by Number of A-List Actors')
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['star_power_impact'] = fig
        return fig
    
    def plot_franchise_analysis(self):
        """Analyze franchise vs non-franchise performance"""
        if 'Is_Franchise' not in self.df.columns or 'Gross_worldwide' not in self.df.columns:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Franchise comparison
        franchise_data = self.df.groupby('Is_Franchise')['Gross_worldwide'].agg(['mean', 'median', 'count']) / 1e6
        
        x = np.arange(len(franchise_data))
        width = 0.35
        
        axes[0].bar(x - width/2, franchise_data['mean'], width, label='Mean', alpha=0.8)
        axes[0].bar(x + width/2, franchise_data['median'], width, label='Median', alpha=0.8)
        axes[0].set_xlabel('Is Franchise')
        axes[0].set_ylabel('Gross (Millions $)')
        axes[0].set_title('Franchise vs Non-Franchise Performance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Non-Franchise', 'Franchise'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Sequel analysis
        if 'Is_Sequel' in self.df.columns:
            sequel_data = self.df.groupby('Is_Sequel')['Gross_worldwide'].agg(['mean', 'median']) / 1e6
            
            x2 = np.arange(len(sequel_data))
            axes[1].bar(x2 - width/2, sequel_data['mean'], width, label='Mean', alpha=0.8)
            axes[1].bar(x2 + width/2, sequel_data['median'], width, label='Median', alpha=0.8)
            axes[1].set_xlabel('Is Sequel')
            axes[1].set_ylabel('Gross (Millions $)')
            axes[1].set_title('Sequel vs Original Performance')
            axes[1].set_xticks(x2)
            axes[1].set_xticklabels(['Original', 'Sequel'])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figures['franchise_analysis'] = fig
        return fig
    
    def generate_all_plots(self, save_path='demo/plots/'):
        """Generate all visualizations (25+ plots)"""
        import os
        
        # Create subdirectories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}distributions/', exist_ok=True)
        os.makedirs(f'{save_path}relationships/', exist_ok=True)
        os.makedirs(f'{save_path}categorical/', exist_ok=True)
        os.makedirs(f'{save_path}temporal/', exist_ok=True)
        os.makedirs(f'{save_path}advanced/', exist_ok=True)
        
        print("\n=== GENERATING 25+ VISUALIZATIONS ===")
        
        self.load_data()
        self.generate_summary_stats()
        
        plot_count = 0
        
        # Original plots
        print("\n[Basic Plots]")
        plot_count += 1
        print(f"{plot_count}. Gross distribution...")
        self.plot_gross_distribution()
        plt.savefig(f'{save_path}gross_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Budget vs Gross...")
        self.plot_budget_vs_gross()
        plt.savefig(f'{save_path}relationships/budget_vs_gross.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Genre analysis...")
        self.plot_genre_analysis()
        plt.savefig(f'{save_path}categorical/genre_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Rating analysis...")
        self.plot_rating_analysis()
        plt.savefig(f'{save_path}rating_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Correlation heatmap...")
        self.plot_correlation_heatmap()
        plt.savefig(f'{save_path}correlation_heatmap.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. ROI analysis...")
        self.plot_roi_analysis()
        plt.savefig(f'{save_path}roi_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Advanced distribution plots
        print("\n[Distribution Plots]")
        for col in ['Budget', 'Gross_worldwide']:
            if col in self.df.columns:
                plot_count += 1
                print(f"{plot_count}. {col} distribution with KDE...")
                self.plot_distribution_with_kde(col, log_scale=True)
                plt.savefig(f'{save_path}distributions/{col}_kde.png', dpi=100, bbox_inches='tight')
                plt.close()
                
                plot_count += 1
                print(f"{plot_count}. {col} Q-Q plot...")
                self.plot_qq(col)
                plt.savefig(f'{save_path}distributions/{col}_qq.png', dpi=100, bbox_inches='tight')
                plt.close()
        
        # Pairwise scatter
        print("\n[Relationship Plots]")
        plot_count += 1
        print(f"{plot_count}. Pairwise scatter matrix...")
        scatter_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating']
        available = [c for c in scatter_cols if c in self.df.columns]
        if len(available) >= 3:
            self.plot_pairwise_scatter(available[:4])
            plt.savefig(f'{save_path}relationships/pairwise_scatter.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # Categorical comparisons
        print("\n[Categorical Plots]")
        if 'Primary_Genre' in self.df.columns and 'Gross_worldwide' in self.df.columns:
            plot_count += 1
            print(f"{plot_count}. Genre vs Gross comparison...")
            self.plot_categorical_comparison('Primary_Genre', 'Gross_worldwide')
            plt.savefig(f'{save_path}categorical/genre_gross_comparison.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # Time trends
        print("\n[Temporal Plots]")
        plot_count += 1
        print(f"{plot_count}. Time trends...")
        self.plot_time_trends()
        plt.savefig(f'{save_path}temporal/time_trends.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Seasonal patterns...")
        self.plot_seasonal_patterns()
        plt.savefig(f'{save_path}temporal/seasonal_patterns.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Advanced analyses
        print("\n[Advanced Analysis Plots]")
        plot_count += 1
        print(f"{plot_count}. Budget tier analysis...")
        self.plot_budget_tier_analysis()
        plt.savefig(f'{save_path}advanced/budget_tier_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Star power impact...")
        self.plot_star_power_impact()
        plt.savefig(f'{save_path}advanced/star_power_impact.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        print(f"{plot_count}. Franchise analysis...")
        self.plot_franchise_analysis()
        plt.savefig(f'{save_path}advanced/franchise_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Generated {plot_count} visualizations")
        print(f"✓ All plots saved to {save_path}")
        plt.close('all')
        
        return self


if __name__ == "__main__":
    visualizer = EDAVisualizer()
    visualizer.generate_all_plots()
