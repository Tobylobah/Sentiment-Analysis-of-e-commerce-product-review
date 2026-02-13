import pandas as pd
import re
import spacy
from datetime import datetime
import os
import dotenv

# --- Load environment variables from .env file ---
dotenv.load_dotenv()

# --- Initialize spaCy NLP model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Automatically download if not found
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class JumiaPreprocessor:
    """
    Preprocessing pipeline for Jumia Nigeria product reviews.
    - Loads multiple category CSVs from environment variable
    - Cleans, normalizes, and enriches text data for ML/NLP tasks
    """

    def __init__(self, input_files):
        """
        input_files: list of paths to CSVs (from .env FILES variable)
        Example in .env:
        FILES=data/jumia_reviews_mobile_phones.csv,data/jumia_reviews_computing.csv,data/jumia_reviews_electronics.csv
        """
        self.input_files = input_files
        self.df = None

    def load_and_merge(self):
        """Load and merge multiple category files into one dataframe."""
        frames = []
        for file in self.input_files:
            file = file.strip()
            if not os.path.exists(file):
                print(f"[WARN] File not found: {file}")
                continue
            temp_df = pd.read_csv(file)
            frames.append(temp_df)

        if not frames:
            raise FileNotFoundError("No valid CSV files found in FILES environment variable.")

        self.df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Loaded {len(self.df)} total raw records from {len(frames)} files.")

    def clean_structure(self):
        """Handle duplicates, missing values, and basic structure cleanup."""
        initial_count = len(self.df)

        # Remove exact duplicates
        self.df.drop_duplicates(inplace=True)

        # Drop rows missing review text
        self.df.dropna(subset=['Review_Text'], inplace=True)

        # Fill missing usernames
        self.df['User_Name'] = self.df['User_Name'].fillna('Anonymous')

        print(f"[INFO] Cleaned structure. Removed {initial_count - len(self.df)} duplicates or empty reviews.")

    def format_features(self):
        """Standardize data types and prepare for modeling."""
        # 1. Convert ratings to integer (handle string values like '5 out of 5')
        self.df['Rating'] = (
            self.df['Rating']
            .astype(str)
            .str.extract(r'(\d+)')  # extract numeric part
            .fillna(0)
            .astype(int)
        )

        # 2. Convert verified badge text to Boolean
        self.df['Verified_Badge'] = self.df['Verified_Badge'].apply(
            lambda x: True if str(x).lower() in ['true', '1', 'verified purchase'] else False
        )

        # 3. Parse timestamp column (Jumia uses DD-MM-YYYY)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], dayfirst=True, errors='coerce')
        self.df.dropna(subset=['Timestamp'], inplace=True)

        print("[INFO] Formatted features: Ratings -> int, Verified_Badge -> bool, Timestamp -> datetime.")

    def normalize_text(self, text):
        """Apply deep text normalization for sentiment/NLP analysis."""
        if not isinstance(text, str):
            return ""

        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # Reduce long character repetitions (e.g., 'goooood' → 'good')
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        # Lemmatize and remove stopwords using spaCy
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
        return " ".join(tokens)

    def process_nlp(self):
        """Run text normalization and create derived text metrics."""
        print("[INFO] Starting NLP normalization — this may take several minutes...")

        # Pre-NLP text length metrics
        self.df['Text_Length'] = self.df['Review_Text'].apply(lambda x: len(str(x)))
        self.df['Word_Count'] = self.df['Review_Text'].apply(lambda x: len(str(x).split()))

        # Normalize text content
        self.df['Cleaned_Text'] = self.df['Review_Text'].apply(self.normalize_text)

        print("[INFO] NLP normalization complete. Added 'Text_Length', 'Word_Count', and 'Cleaned_Text'.")

    def save_data(self, output_name="cleaned_jumia_reviews.csv"):
        """Save the fully processed dataset."""
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", output_name)
        self.df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    # --- Load CSV file list from .env variable ---
    raw_files = os.getenv("FILES")  # e.g. "data/jumia_reviews_mobile_phones.csv,data/jumia_reviews_computing.csv"
    if not raw_files:
        raise ValueError("FILES variable not found in environment. Please define it in your .env file.")

    # Convert comma-separated list to Python list
    files = [f.strip() for f in raw_files.split(",") if f.strip()]

    # --- Run the pipeline ---
    preprocessor = JumiaPreprocessor(files)
    preprocessor.load_and_merge()
    preprocessor.clean_structure()
    preprocessor.format_features()
    preprocessor.process_nlp()
    preprocessor.save_data()
