"""
Advanced Feature Engineering for Box Office Prediction
Transforms raw movie data into ML-ready features using domain knowledge
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import re
import warnings
warnings.filterwarnings('ignore')

# ==================== LOOKUP TABLES ====================

# Top grossing actors (based on box office history)
A_LIST_ACTORS = [
    "Leonardo DiCaprio", "Tom Cruise", "Dwayne Johnson", "Scarlett Johansson",
    "Robert Downey Jr.", "Tom Hanks", "Brad Pitt", "Johnny Depp", "Will Smith",
    "Angelina Jolie", "Samuel L. Jackson", "Chris Hemsworth", "Chris Evans",
    "Chris Pratt", "Jennifer Lawrence", "Mark Ruffalo", "Christian Bale",
    "Matt Damon", "Harrison Ford", "Morgan Freeman", "Keanu Reeves",
    "Ryan Reynolds", "Vin Diesel", "Margot Robbie", "Zendaya", "Tom Holland",
    "Benedict Cumberbatch", "Cillian Murphy", "Timothée Chalamet",
    "Robert Pattinson", "Hugh Jackman", "Jake Gyllenhaal", "Ryan Gosling"
]

# Top grossing directors
A_LIST_DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "James Cameron",
    "Denis Villeneuve", "Quentin Tarantino", "Ridley Scott", "Peter Jackson",
    "David Fincher", "Francis Ford Coppola", "George Lucas", "Stanley Kubrick",
    "Alfred Hitchcock", "Clint Eastwood", "Zack Snyder", "Jon Watts",
    "Joss Whedon", "Anthony Russo", "Joe Russo", "J.J. Abrams", "Michael Bay",
    "Sam Raimi", "Taika Waititi", "Greta Gerwig", "Jordan Peele", "Matt Reeves"
]

# Major Hollywood studios
MAJOR_STUDIOS = [
    "Walt Disney", "Disney", "Warner Bros", "Universal", "Paramount",
    "Sony", "Columbia", "20th Century", "Marvel Studios", "Lionsgate",
    "New Line Cinema", "DreamWorks", "MGM", "Fox"
]

# Known franchises and keywords
FRANCHISE_KEYWORDS = [
    'marvel', 'dc', 'star wars', 'harry potter', 'fast furious', 'avengers',
    'x-men', 'spider-man', 'batman', 'superman', 'james bond', 'mission impossible',
    'jurassic', 'transformers', 'lord of the rings', 'hobbit', 'pirates caribbean',
    'toy story', 'frozen', 'incredibles', 'finding', 'cars', 'shrek',
    'star trek', 'mad max', 'indiana jones', 'alien', 'terminator', 'rocky',
    'john wick', 'matrix', 'hunger games', 'twilight', 'conjuring'
]

# Sequel indicators
SEQUEL_INDICATORS = [
    r'\b(II|III|IV|V|VI|VII|VIII|IX|X)\b',  # Roman numerals
    r'\b\d\b',  # Single digits (e.g., "Movie 2")
    r'Part \d',  # "Part 2"
    r'Chapter \d',  # "Chapter 2"
    r'Returns?',  # "Returns"
    r'Reloaded',  # "Reloaded"
    r'Resurrections?',  # "Resurrections"
    r'Reborn',  # "Reborn"
]


class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_metadata = {}
        print(f"FeatureEngineer initialized with {len(df)} movies")
    
    # ==================== TEMPORAL FEATURES ====================
    
    def extract_temporal_features(self) -> 'FeatureEngineer':
        """Extract all time-based features"""
        print("\n[1/8] Extracting temporal features...")
        
        # Ensure Release_Data is datetime
        self.df['Release_Data_dt'] = pd.to_datetime(self.df['Release_Data'], errors='coerce')
        
        # Release Month (1-12)
        self.df['Release_Month'] = self.df['Release_Data_dt'].dt.month
        
        # Release Quarter (Q1-Q4)
        self.df['Release_Quarter'] = self.df['Release_Data_dt'].dt.quarter
        
        # Release Day of Week (0=Monday, 6=Sunday)
        self.df['Release_DayOfWeek'] = self.df['Release_Data_dt'].dt.dayofweek
        
        # Boolean: Is Summer Release (May-August)
        self.df['Is_Summer_Release'] = self.df['Release_Month'].apply(
            lambda x: 1 if x in [5, 6, 7, 8] else 0 if pd.notna(x) else np.nan
        )
        
        # Boolean: Is Holiday Release (Nov-Dec)
        self.df['Is_Holiday_Release'] = self.df['Release_Month'].apply(
            lambda x: 1 if x in [11, 12] else 0 if pd.notna(x) else np.nan
        )
        
        # Boolean: Is Awards Season (Jan-Feb)
        self.df['Is_Awards_Season'] = self.df['Release_Month'].apply(
            lambda x: 1 if x in [1, 2] else 0 if pd.notna(x) else np.nan
        )
        
        # Decade
        self.df['Decade'] = (self.df['Release_Year'] // 10 * 10).astype('Int64')
        
        # Movie Age (years since release)
        current_year = pd.Timestamp.now().year
        self.df['Movie_Age'] = current_year - self.df['Release_Year']
        
        print(f"  ✓ Created 8 temporal features")
        return self
    
    # ==================== BUDGET FEATURES ====================
    
    def create_budget_features(self) -> 'FeatureEngineer':
        """Create budget tiers and log transforms"""
        print("\n[2/8] Creating budget features...")
        
        if 'Budget' not in self.df.columns:
            print("  ⚠ Budget column not found")
            return self
        
        # Log Budget (handle zeros)
        self.df['Log_Budget'] = np.log10(self.df['Budget'].replace(0, np.nan))
        
        # Budget Tier (categorical)
        def categorize_budget(budget):
            if pd.isna(budget):
                return np.nan
            if budget < 5_000_000:
                return 'Micro'
            elif budget < 25_000_000:
                return 'Low'
            elif budget < 75_000_000:
                return 'Medium'
            elif budget < 150_000_000:
                return 'High'
            else:
                return 'Blockbuster'
        
        self.df['Budget_Tier'] = self.df['Budget'].apply(categorize_budget)
        
        # Is High Budget (>$100M)
        self.df['Is_High_Budget'] = (self.df['Budget'] > 100_000_000).astype(int)
        
        # Budget Percentile
        self.df['Budget_Percentile'] = self.df['Budget'].rank(pct=True) * 100
        
        # Budget Per Genre (relative to genre median)
        if 'Primary_Genre' in self.df.columns:
            genre_budget_median = self.df.groupby('Primary_Genre')['Budget'].transform('median')
            self.df['Budget_Per_Genre'] = self.df['Budget'] / genre_budget_median
        
        print(f"  ✓ Created 5 budget features")
        return self
    
    # ==================== CONTENT/GENRE FEATURES ====================
    
    def create_genre_features(self) -> 'FeatureEngineer':
        """Binary encode top genres + genre popularity score"""
        print("\n[3/8] Creating content/genre features...")
        
        if 'Genre' not in self.df.columns:
            print("  ⚠ Genre column not found")
            return self
        
        # Create binary columns for major genres
        major_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Horror', 'Animation']
        
        for genre in major_genres:
            col_name = f'Is_{genre}'
            self.df[col_name] = self.df['Genre'].fillna('').str.contains(genre, case=False, regex=False).astype(int)
        
        # Sci-Fi (includes Sci-Fi, Fantasy, Adventure)
        self.df['Is_SciFi'] = (
            self.df['Genre'].fillna('').str.contains('Sci-Fi|Fantasy', case=False, regex=True)
        ).astype(int)
        
        # Genre Popularity Score (historical avg gross per genre)
        if 'Gross_worldwide' in self.df.columns and 'Primary_Genre' in self.df.columns:
            genre_avg_gross = self.df.groupby('Primary_Genre')['Gross_worldwide'].transform('mean')
            self.df['Genre_Popularity_Score'] = genre_avg_gross
        
        print(f"  ✓ Created 8 genre features")
        return self
    
    def detect_sequel_franchise(self) -> 'FeatureEngineer':
        """Detect sequels and known franchises"""
        print("\n[4/8] Detecting sequels and franchises...")
        
        if 'Movie_Title' not in self.df.columns:
            print("  ⚠ Movie_Title column not found")
            return self
        
        # Is Sequel
        def is_sequel(title):
            if pd.isna(title):
                return 0
            title = str(title)
            for pattern in SEQUEL_INDICATORS:
                if re.search(pattern, title, re.IGNORECASE):
                    return 1
            return 0
        
        self.df['Is_Sequel'] = self.df['Movie_Title'].apply(is_sequel)
        
        # Is Franchise
        def is_franchise(row):
            title = str(row.get('Movie_Title', '')).lower()
            keywords = str(row.get('Keywords', '')).lower()
            combined = title + ' ' + keywords
            
            for franchise in FRANCHISE_KEYWORDS:
                if franchise in combined:
                    return 1
            return 0
        
        self.df['Is_Franchise'] = self.df.apply(is_franchise, axis=1)
        
        # Is Adaptation (based on novel/comic)
        if 'Keywords' in self.df.columns:
            self.df['Is_Adaptation'] = self.df['Keywords'].fillna('').str.contains(
                'based on novel|based on comic|based on book|based on the works of', 
                case=False, regex=True
            ).astype(int)
        else:
            self.df['Is_Adaptation'] = 0
        
        # Has Superhero
        if 'Keywords' in self.df.columns:
            self.df['Has_Superhero'] = self.df['Keywords'].fillna('').str.contains(
                'superhero|super hero|batman|superman|spider-man|iron man|captain america',
                case=False, regex=True
            ).astype(int)
        else:
            self.df['Has_Superhero'] = 0
        
        print(f"  ✓ Created 4 content detection features")
        return self
    
    # ==================== STAR POWER FEATURES ====================
    
    def calculate_star_power(self) -> 'FeatureEngineer':
        """Score movies by cast/director star power"""
        print("\n[5/8] Calculating star power...")
        
        # Has A-List Actor
        def count_a_list_actors(cast_str):
            if pd.isna(cast_str) or cast_str == '':
                return 0
            cast_list = [c.strip() for c in str(cast_str).split(',')]
            return sum(1 for actor in cast_list if actor in A_LIST_ACTORS)
        
        if 'Cast' in self.df.columns:
            self.df['Top_Actor_Count'] = self.df['Cast'].apply(count_a_list_actors)
            self.df['Has_A_List_Actor'] = (self.df['Top_Actor_Count'] > 0).astype(int)
        else:
            self.df['Top_Actor_Count'] = 0
            self.df['Has_A_List_Actor'] = 0
        
        # Has A-List Director
        def has_a_list_director(crew_str):
            if pd.isna(crew_str) or crew_str == '':
                return 0
            crew_list = [c.strip() for c in str(crew_str).split(',')]
            return int(any(director in crew_list for director in A_LIST_DIRECTORS))
        
        if 'Crew' in self.df.columns:
            self.df['Has_A_List_Director'] = self.df['Crew'].apply(has_a_list_director)
        else:
            self.df['Has_A_List_Director'] = 0
        
        # Director Average Gross (if we have historical data - simplified version)
        if 'Crew' in self.df.columns and 'Gross_worldwide' in self.df.columns:
            # Extract first director
            self.df['First_Director'] = self.df['Crew'].fillna('').apply(
                lambda x: x.split(',')[0].strip() if x else 'Unknown'
            )
            director_avg = self.df.groupby('First_Director')['Gross_worldwide'].transform('mean')
            self.df['Director_Avg_Gross'] = director_avg
        
        # Lead Actor Average Gross (simplified)
        if 'Cast' in self.df.columns and 'Gross_worldwide' in self.df.columns:
            self.df['Lead_Actor'] = self.df['Cast'].fillna('').apply(
                lambda x: x.split(',')[0].strip() if x else 'Unknown'
            )
            actor_avg = self.df.groupby('Lead_Actor')['Gross_worldwide'].transform('mean')
            self.df['Lead_Actor_Avg_Gross'] = actor_avg
        
        print(f"  ✓ Created 6 star power features")
        return self
    
    # ==================== STUDIO FEATURES ====================
    
    def create_studio_features(self) -> 'FeatureEngineer':
        """Studio reputation scoring"""
        print("\n[6/8] Creating studio features...")
        
        if 'Studios' not in self.df.columns:
            print("  ⚠ Studios column not found")
            return self
        
        # Is Major Studio
        def is_major_studio(studios_str):
            if pd.isna(studios_str) or studios_str == '':
                return 0
            studios_lower = str(studios_str).lower()
            return int(any(studio.lower() in studios_lower for studio in MAJOR_STUDIOS))
        
        self.df['Is_Major_Studio'] = self.df['Studios'].apply(is_major_studio)
        
        # Studio Average Gross
        if 'Gross_worldwide' in self.df.columns:
            self.df['Primary_Studio'] = self.df['Studios'].fillna('').apply(
                lambda x: x.split(',')[0].strip() if x else 'Unknown'
            )
            studio_avg = self.df.groupby('Primary_Studio')['Gross_worldwide'].transform('mean')
            self.df['Studio_Avg_Gross'] = studio_avg
        
        print(f"  ✓ Created 3 studio features")
        return self
    
    # ==================== RATING/CERTIFICATION FEATURES ====================
    
    def create_rating_features(self) -> 'FeatureEngineer':
        """MPAA encoding and rating buckets"""
        print("\n[7/8] Creating rating/certification features...")
        
        # MPAA Encoded (if ListOfCertificate exists)
        if 'ListOfCertificate' in self.df.columns:
            mpaa_mapping = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4, 'UNRATED': 2}
            
            def encode_mpaa(cert_str):
                if pd.isna(cert_str) or cert_str == '':
                    return np.nan
                # Extract first certification
                cert = str(cert_str).split(',')[0].strip()
                return mpaa_mapping.get(cert, 2)  # Default to PG-13 equivalent
            
            self.df['MPAA_Encoded'] = self.df['ListOfCertificate'].apply(encode_mpaa)
            
            # Is Family Friendly (G or PG)
            self.df['Is_Family_Friendly'] = self.df['ListOfCertificate'].fillna('').str.contains(
                r'\b(G|PG)\b', regex=True
            ).astype(int)
            
            # Is Adult Only (R or NC-17)
            self.df['Is_Adult_Only'] = self.df['ListOfCertificate'].fillna('').str.contains(
                'R|NC-17', regex=True
            ).astype(int)
        
        # Rating Bucket (Excellent/Good/Average/Poor)
        if 'Rating' in self.df.columns:
            def rating_bucket(rating):
                if pd.isna(rating):
                    return np.nan
                if rating >= 8.0:
                    return 'Excellent'
                elif rating >= 7.0:
                    return 'Good'
                elif rating >= 5.0:
                    return 'Average'
                else:
                    return 'Poor'
            
            self.df['Rating_Bucket'] = self.df['Rating'].apply(rating_bucket)
            
            # Is Highly Rated
            self.df['Is_Highly_Rated'] = (self.df['Rating'] >= 7.5).astype(int)
        
        # Rating Count Log
        if 'Rating_Count' in self.df.columns:
            self.df['Rating_Count_Log'] = np.log10(self.df['Rating_Count'].replace(0, np.nan))
            
            # Is Popular (>100k ratings)
            self.df['Is_Popular'] = (self.df['Rating_Count'] > 100_000).astype(int)
        
        print(f"  ✓ Created 7 rating features")
        return self
    
    # ==================== GEOGRAPHIC FEATURES ====================
    
    def create_geographic_features(self) -> 'FeatureEngineer':
        """Language and country features"""
        print("\n[8/8] Creating geographic features...")
        
        # Is English
        if 'Languages' in self.df.columns:
            self.df['Is_English'] = self.df['Languages'].fillna('').str.contains(
                'English', case=False
            ).astype(int)
            
            # Is Multilingual
            self.df['Is_Multilingual'] = (self.df['Languages_Count'] > 1).astype(int)
        
        # Is US Production
        if 'Countries' in self.df.columns:
            self.df['Is_US_Production'] = self.df['Countries'].fillna('').str.contains(
                'United States|USA', case=False, regex=True
            ).astype(int)
            
            # Is International Coproduction
            self.df['Is_International_Coproduction'] = (self.df['Countries_Count'] > 1).astype(int)
        
        # Market Reach Score (Languages × Countries)
        if 'Languages_Count' in self.df.columns and 'Countries_Count' in self.df.columns:
            self.df['Market_Reach_Score'] = self.df['Languages_Count'] * self.df['Countries_Count']
        
        print(f"  ✓ Created 6 geographic features")
        return self
    
    # ==================== RATIO FEATURES ====================
    
    def create_ratio_features(self) -> 'FeatureEngineer':
        """Derived ratio features"""
        print("\nCreating ratio features...")
        
        # Budget Runtime Ratio (cost per minute)
        if 'Budget' in self.df.columns and 'Runtime' in self.df.columns:
            self.df['Budget_Runtime_Ratio'] = self.df['Budget'] / self.df['Runtime']
        
        # Cast to Budget Ratio (star efficiency)
        if 'Cast_Count' in self.df.columns and 'Budget' in self.df.columns:
            self.df['Cast_to_Budget_Ratio'] = self.df['Cast_Count'] / (self.df['Budget'] / 1_000_000)
        
        # Keyword Density (marketing intensity)
        if 'Keywords_Count' in self.df.columns and 'Genre_Count' in self.df.columns:
            self.df['Keyword_Density'] = self.df['Keywords_Count'] / self.df['Genre_Count'].replace(0, 1)
        
        # Production Scale
        if 'Countries_Count' in self.df.columns:
            studio_count = self.df['Studios'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
            self.df['Production_Scale'] = studio_count * self.df['Countries_Count']
        
        print(f"  ✓ Created 4 ratio features")
        return self
    
    # ==================== MAIN PIPELINE ====================
    
    def run_full_engineering(self) -> pd.DataFrame:
        """Execute all feature engineering steps"""
        print("\n" + "="*70)
        print(" FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        initial_cols = len(self.df.columns)
        
        (self
            .extract_temporal_features()
            .create_budget_features()
            .create_genre_features()
            .detect_sequel_franchise()
            .calculate_star_power()
            .create_studio_features()
            .create_rating_features()
            .create_geographic_features()
            .create_ratio_features())
        
        final_cols = len(self.df.columns)
        new_features = final_cols - initial_cols
        
        print("\n" + "="*70)
        print(f" FEATURE ENGINEERING COMPLETED")
        print(f" Created {new_features} new features!")
        print(f" Total features: {final_cols}")
        print("="*70)
        
        return self.df
    
    def get_feature_list(self) -> List[str]:
        """Return list of all engineered feature names"""
        # All new feature names
        new_features = [
            # Temporal
            'Release_Month', 'Release_Quarter', 'Release_DayOfWeek', 
            'Is_Summer_Release', 'Is_Holiday_Release', 'Is_Awards_Season', 
            'Decade', 'Movie_Age',
            # Budget
            'Log_Budget', 'Budget_Tier', 'Is_High_Budget', 'Budget_Percentile', 'Budget_Per_Genre',
            # Genre/Content
            'Is_Action', 'Is_Comedy', 'Is_Drama', 'Is_Thriller', 'Is_Horror', 
            'Is_Animation', 'Is_SciFi', 'Genre_Popularity_Score',
            'Is_Sequel', 'Is_Franchise', 'Is_Adaptation', 'Has_Superhero',
            # Star Power
            'Top_Actor_Count', 'Has_A_List_Actor', 'Has_A_List_Director',
            'Director_Avg_Gross', 'Lead_Actor_Avg_Gross',
            # Studio
            'Is_Major_Studio', 'Studio_Avg_Gross',
            # Rating
            'MPAA_Encoded', 'Is_Family_Friendly', 'Is_Adult_Only', 
            'Rating_Bucket', 'Is_Highly_Rated', 'Rating_Count_Log', 'Is_Popular',
            # Geographic
            'Is_English', 'Is_Multilingual', 'Is_US_Production', 
            'Is_International_Coproduction', 'Market_Reach_Score',
            # Ratios
            'Budget_Runtime_Ratio', 'Cast_to_Budget_Ratio', 
            'Keyword_Density', 'Production_Scale'
        ]
        return [f for f in new_features if f in self.df.columns]
    
    def get_feature_metadata(self) -> Dict:
        """Return metadata about each feature for documentation"""
        metadata = {
            'total_features_created': len(self.get_feature_list()),
            'categories': {
                'Temporal': 8,
                'Budget & Financial': 5,
                'Content & Genre': 12,
                'Star Power': 6,
                'Studio': 3,
                'Rating & Certification': 7,
                'Geographic': 6,
                'Derived Ratios': 4
            },
            'feature_list': self.get_feature_list()
        }
        return metadata


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("This module creates 50+ features from raw movie data")
    print("\nUsage:")
    print("  from feature_engineering import FeatureEngineer")
    print("  engineer = FeatureEngineer(df)")
    print("  df_featured = engineer.run_full_engineering()")

