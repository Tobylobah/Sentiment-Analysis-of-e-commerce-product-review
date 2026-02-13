import pandas as pd
import re
import spacy
from datetime import datetime
import os
import dotenv

# Load spaCy for expert-level lemmatization and tokenization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class JumiaPreprocessor:
    def __init__(self, input_files):
        """
        input_files: List of paths to your category CSVs 
        (e.g., ['mobile.csv', 'computing.csv', 'electronics.csv'])
        """
        self.input_files = input_files
        self.df = None

    def load_and_merge(self):
        """Merges multiple category files while preserving the category label.[4, 5]"""
        frames = []
        for file in self.input_files:
            temp_df = pd.read_csv(file)
            frames.append(temp_df)
        self.df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Loaded {len(self.df)} total raw records.")

    def clean_structure(self):
        """Standard data cleaning: deduplication and null handling.[4, 5]"""
        # Remove exact duplicates from scraping retries
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        
        # Essential: Drop rows without review text as they cannot be used for sentiment
        self.df.dropna(subset=['Review_Text'], inplace=True)
        
        # Fill missing usernames with 'Anonymous' to track reviewer history [1, 6]
        self.df['User_Name'] = self.df['User_Name'].fillna('Anonymous')
        
        print(f"[INFO] Cleaned structure. Removed {initial_count - len(self.df)} records.")

    def format_features(self):
        """Standardizes data types for mathematical modeling.[4, 7]"""
        # 1. Rating: Ensure discrete integer 1-5 [7, 8]
        self.df = pd.to_numeric(self.df, errors='coerce').fillna(0).astype(int)
        
        # 2. Verified Badge: Convert to Boolean [6, 3]
        # Jumia badges often appear as "Verified Purchase" string or a specific class presence
        self.df = self.df.apply(
            lambda x: True if str(x).lower() in ['true', '1', 'verified purchase'] else False
        )

        # 3. Timestamp: Critical for Burst Detection [1, 3]
        # Jumia Nigeria typically uses DD-MM-YYYY
        self.df = pd.to_datetime(self.df, dayfirst=True, errors='coerce')
        self.df.dropna(subset=['Timestamp'], inplace=True) # Remove rows with invalid dates

    def normalize_text(self, text):
        """Advanced NLP cleaning for the Nigerian e-commerce context.[9, 2]"""
        if not isinstance(text, str): return ""
        
        # Lowercase and remove noise
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation [9]
        
        # Reduce character repetition (e.g., "greeeeat" -> "great") [2]
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Lemmatization (reducing words to their root form)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
        return " ".join(tokens)

    def process_nlp(self):
        """Applies normalization and creates analytical metrics.[9, 3]"""
        print("[INFO] Starting NLP normalization (this may take a minute)...")
        # Create metrics before cleaning text to preserve original 'effort' indicators
        self.df = self.df.apply(len)
        self.df = self.df.apply(lambda x: len(str(x).split()))
        
        # Apply the deep cleaning
        self.df = self.df.apply(self.normalize_text)

    def save_data(self, output_name="cleaned_jumia_reviews.csv"):
        self.df.to_csv(output_name, index=False)
        print(f" Preprocessed data saved to {output_name}")

if __name__ == "__main__":
    # Update these filenames to match your scraped output
    files = os.getenv("FILES").split(", ")S
    
    preprocessor = JumiaPreprocessor(files)
    preprocessor.load_and_merge()
    preprocessor.clean_structure()
    preprocessor.format_features()
    preprocessor.process_nlp()
    preprocessor.save_data()