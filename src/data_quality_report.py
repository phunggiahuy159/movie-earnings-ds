"""
Data Quality Report Generator
Checks for: Missing values, duplicates, data types, distributions
Generates comprehensive HTML report for data quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """Comprehensive data quality validation and reporting"""
    
    def __init__(self, data_path='dataset/data_joined.csv'):
        self.data_path = data_path
        self.df = None
        self.report_lines = []
        
    def load_data(self):
        """Load raw data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} movies with {len(self.df.columns)} columns")
        return self
    
    def check_basic_info(self):
        """Generate basic dataset information"""
        print("\n=== BASIC DATASET INFO ===")
        info = {
            'Total Movies': len(self.df),
            'Total Columns': len(self.df.columns),
            'Memory Usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        self.report_lines.append("<h2>📊 Basic Dataset Information</h2>")
        self.report_lines.append("<table border='1' style='border-collapse: collapse; width: 50%;'>")
        for key, value in info.items():
            self.report_lines.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
            print(f"{key}: {value}")
        self.report_lines.append("</table><br>")
        
        return self
    
    def check_missing_values(self):
        """Generate missing value report per column"""
        print("\n=== MISSING VALUES ===")
        
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percent': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        }).sort_values('Missing_Percent', ascending=False)
        
        # Filter columns with missing values
        missing_with_nulls = missing[missing['Missing_Count'] > 0]
        
        self.report_lines.append("<h2>🔍 Missing Values Analysis</h2>")
        if len(missing_with_nulls) > 0:
            self.report_lines.append(f"<p><strong>Columns with missing data: {len(missing_with_nulls)}</strong></p>")
            self.report_lines.append(missing_with_nulls.to_html(index=False))
            print(missing_with_nulls.to_string(index=False))
        else:
            self.report_lines.append("<p>✓ No missing values found!</p>")
            print("✓ No missing values found!")
        
        self.report_lines.append("<br>")
        return self
    
    def check_duplicates(self):
        """Find duplicate Movie_IDs"""
        print("\n=== DUPLICATE CHECK ===")
        
        if 'Movie_ID' in self.df.columns:
            duplicates = self.df['Movie_ID'].duplicated().sum()
            unique_ids = self.df['Movie_ID'].nunique()
            
            self.report_lines.append("<h2>🔄 Duplicate Analysis</h2>")
            self.report_lines.append(f"<p><strong>Total Movies:</strong> {len(self.df)}</p>")
            self.report_lines.append(f"<p><strong>Unique Movie IDs:</strong> {unique_ids}</p>")
            self.report_lines.append(f"<p><strong>Duplicate IDs:</strong> {duplicates}</p>")
            
            if duplicates > 0:
                self.report_lines.append(f"<p style='color: red;'>⚠️ Warning: {duplicates} duplicate Movie_IDs found!</p>")
                print(f"⚠️ Warning: {duplicates} duplicate Movie_IDs found!")
            else:
                self.report_lines.append("<p style='color: green;'>✓ No duplicate Movie_IDs</p>")
                print("✓ No duplicate Movie_IDs")
        else:
            self.report_lines.append("<p>⚠️ No Movie_ID column found</p>")
            print("⚠️ No Movie_ID column found")
        
        self.report_lines.append("<br>")
        return self
    
    def check_data_types(self):
        """Validate expected data types"""
        print("\n=== DATA TYPES ===")
        
        dtypes = pd.DataFrame({
            'Column': self.df.dtypes.index,
            'Type': self.df.dtypes.values.astype(str)
        })
        
        self.report_lines.append("<h2>📝 Data Types</h2>")
        self.report_lines.append(dtypes.to_html(index=False))
        print(dtypes.to_string(index=False))
        
        self.report_lines.append("<br>")
        return self
    
    def check_value_ranges(self):
        """Check for unrealistic values (negative budgets, etc.)"""
        print("\n=== VALUE RANGE VALIDATION ===")
        
        self.report_lines.append("<h2>📏 Value Range Validation</h2>")
        issues = []
        
        # Check Budget
        if 'Budget' in self.df.columns:
            budget_numeric = pd.to_numeric(self.df['Budget'], errors='coerce')
            negative_budgets = (budget_numeric < 0).sum()
            if negative_budgets > 0:
                issues.append(f"⚠️ {negative_budgets} movies with negative Budget")
                print(f"⚠️ {negative_budgets} movies with negative Budget")
        
        # Check Gross_worldwide
        if 'Gross_worldwide' in self.df.columns:
            gross_numeric = pd.to_numeric(self.df['Gross_worldwide'], errors='coerce')
            negative_gross = (gross_numeric < 0).sum()
            if negative_gross > 0:
                issues.append(f"⚠️ {negative_gross} movies with negative Gross")
                print(f"⚠️ {negative_gross} movies with negative Gross")
        
        # Check Rating
        if 'Rating' in self.df.columns:
            rating_numeric = pd.to_numeric(self.df['Rating'], errors='coerce')
            invalid_ratings = ((rating_numeric < 0) | (rating_numeric > 10)).sum()
            if invalid_ratings > 0:
                issues.append(f"⚠️ {invalid_ratings} movies with Rating outside 0-10 range")
                print(f"⚠️ {invalid_ratings} movies with Rating outside 0-10 range")
        
        # Check Release_Data for future dates
        if 'Release_Data' in self.df.columns:
            try:
                release_dates = pd.to_datetime(self.df['Release_Data'], errors='coerce')
                future_dates = (release_dates > pd.Timestamp.now()).sum()
                if future_dates > 0:
                    issues.append(f"⚠️ {future_dates} movies with future release dates")
                    print(f"⚠️ {future_dates} movies with future release dates")
            except:
                pass
        
        if issues:
            self.report_lines.append("<ul>")
            for issue in issues:
                self.report_lines.append(f"<li>{issue}</li>")
            self.report_lines.append("</ul>")
        else:
            self.report_lines.append("<p style='color: green;'>✓ All value ranges are valid</p>")
            print("✓ All value ranges are valid")
        
        self.report_lines.append("<br>")
        return self
    
    def check_numeric_distributions(self):
        """Check distributions of numeric columns"""
        print("\n=== NUMERIC DISTRIBUTIONS ===")
        
        numeric_cols = ['Budget', 'Gross_worldwide', 'Runtime', 'Rating', 'Rating_Count']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if not available_cols:
            return self
        
        self.report_lines.append("<h2>📈 Numeric Column Statistics</h2>")
        
        stats_data = []
        for col in available_cols:
            col_numeric = pd.to_numeric(self.df[col], errors='coerce')
            col_clean = col_numeric.dropna()
            
            if len(col_clean) > 0:
                stats_data.append({
                    'Column': col,
                    'Count': len(col_clean),
                    'Mean': f"{col_clean.mean():.2f}",
                    'Median': f"{col_clean.median():.2f}",
                    'Std': f"{col_clean.std():.2f}",
                    'Min': f"{col_clean.min():.2f}",
                    'Max': f"{col_clean.max():.2f}"
                })
        
        stats_df = pd.DataFrame(stats_data)
        self.report_lines.append(stats_df.to_html(index=False))
        print(stats_df.to_string(index=False))
        
        self.report_lines.append("<br>")
        return self
    
    def check_categorical_distributions(self):
        """Check categorical column distributions"""
        print("\n=== CATEGORICAL DISTRIBUTIONS ===")
        
        categorical_cols = ['Genre', 'Primary_Genre', 'Studios', 'Languages', 'Countries']
        available_cols = [col for col in categorical_cols if col in self.df.columns]
        
        if not available_cols:
            return self
        
        self.report_lines.append("<h2>📊 Categorical Column Distributions</h2>")
        
        for col in available_cols:
            unique_count = self.df[col].nunique()
            top_5 = self.df[col].value_counts().head(5)
            
            self.report_lines.append(f"<h3>{col}</h3>")
            self.report_lines.append(f"<p><strong>Unique values:</strong> {unique_count}</p>")
            self.report_lines.append(f"<p><strong>Top 5 most common:</strong></p>")
            self.report_lines.append(top_5.to_frame('Count').to_html())
            
            print(f"\n{col}: {unique_count} unique values")
            print(top_5)
        
        self.report_lines.append("<br>")
        return self
    
    def generate_report(self, output_path='demo/data_quality_report.html'):
        """Export full HTML report"""
        print(f"\n=== GENERATING REPORT ===")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build HTML (with escaped CSS braces)
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        p {{
            line-height: 1.6;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🎬 Movie Dataset - Data Quality Report</h1>
    <p><strong>Generated:</strong> {date}</p>
    <p><strong>Dataset:</strong> {dataset_path}</p>
    <hr>
    
    {content}
    
    <div class="footer">
        <p>Generated by Data Quality Checker | Movie Box Office Prediction Project</p>
    </div>
</body>
</html>
""".format(
            date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset_path=self.data_path,
            content='\n'.join(self.report_lines)
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Report saved to: {output_path}")
        return self
    
    def run_full_check(self):
        """Execute all quality checks"""
        print("\n" + "="*70)
        print(" DATA QUALITY VALIDATION")
        print("="*70)
        
        (self
            .load_data()
            .check_basic_info()
            .check_missing_values()
            .check_duplicates()
            .check_data_types()
            .check_value_ranges()
            .check_numeric_distributions()
            .check_categorical_distributions()
            .generate_report())
        
        print("\n" + "="*70)
        print(" DATA QUALITY CHECK COMPLETED")
        print("="*70)
        
        return self


if __name__ == "__main__":
    checker = DataQualityChecker('dataset/data_joined.csv')
    checker.run_full_check()

