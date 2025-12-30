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
    
    def generate_all_plots(self, save_path='demo/plots/'):
        """Generate all visualizations"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        self.load_data()
        self.generate_summary_stats()
        
        print("\n1. Gross distribution...")
        self.plot_gross_distribution()
        plt.savefig(f'{save_path}gross_distribution.png', dpi=100, bbox_inches='tight')
        
        print("2. Budget vs Gross...")
        self.plot_budget_vs_gross()
        plt.savefig(f'{save_path}budget_vs_gross.png', dpi=100, bbox_inches='tight')
        
        print("3. Genre analysis...")
        self.plot_genre_analysis()
        plt.savefig(f'{save_path}genre_analysis.png', dpi=100, bbox_inches='tight')
        
        print("4. Rating analysis...")
        self.plot_rating_analysis()
        plt.savefig(f'{save_path}rating_analysis.png', dpi=100, bbox_inches='tight')
        
        print("5. Correlation heatmap...")
        self.plot_correlation_heatmap()
        plt.savefig(f'{save_path}correlation_heatmap.png', dpi=100, bbox_inches='tight')
        
        print("6. ROI analysis...")
        self.plot_roi_analysis()
        plt.savefig(f'{save_path}roi_analysis.png', dpi=100, bbox_inches='tight')
        
        print(f"\nAll plots saved to {save_path}")
        plt.close('all')
        
        return self


if __name__ == "__main__":
    visualizer = EDAVisualizer()
    visualizer.generate_all_plots()
