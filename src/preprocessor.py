import os
import re
import json
import hashlib
import warnings
from typing import List, Dict

import dotenv
import pandas as pd
import numpy as np

dotenv.load_dotenv()
warnings.filterwarnings("ignore")


class JumiaPreprocessor:
    """
    End-to-end preprocessing for Jumia review fraud + sentiment project.

    Responsibilities:
    - Load raw CSVs
    - Clean structure
    - Normalize Nigerian/Pidgin text
    - Deduplicate exact and near duplicates
    - Generate behavioural features needed downstream
    - Create proxy labels for deceptive/genuine reviews
    """

    PIDGIN_MAP: Dict[str, str] = {
        "bcos": "because",
        "becos": "because",
        "bikos": "because",
        "abt": "about",
        "u": "you",
        "ur": "your",
        "una": "you_all",
        "dey": "dey",
        "no": "not",
        "nor": "not",
        "e": "it",
        "wahala": "problem",
        "sharpaly": "quickly",
        "gud": "good",
        "luv": "love",
        "f9": "fine",
        "nd": "and",
        "dis": "this",
        "dat": "that",
        "diz": "this",
        "dem": "them",
        "de": "the",
        "wot": "what",
        "wetin": "what",
        "abi": "right",
        "sef": "self",
        "na": "is",
        "pls": "please",
        "pls.": "please",
        "tnx": "thanks",
        "thx": "thanks",
        "wk": "week",
        "yr": "year",
        "pikin": "child",
        "greeeaat": "great"
    }

    CRITICAL_NEGATORS = {"not", "no", "never", "dey", "nor", "nope", "hardly"}

    def __init__(self, input_files: List[str]):
        self.input_files = input_files
        self.df: pd.DataFrame | None = None

    def load_and_merge(self) -> pd.DataFrame:
        frames = []
        for file in self.input_files:
            file = file.strip()
            if not os.path.exists(file):
                print(f"[WARN] Missing file: {file}")
                continue
            temp = pd.read_csv(file)
            temp["Source_File"] = os.path.basename(file)
            frames.append(temp)

        if not frames:
            raise FileNotFoundError("No raw CSV files were found.")

        self.df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Loaded {len(self.df)} raw records from {len(frames)} files.")
        return self.df

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        expected = [
            "Category", "Product_URL", "User_Name", "Rating",
            "Review_Title", "Review_Text", "Timestamp", "Verified_Badge"
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def _parse_rating(x) -> int:
        if pd.isna(x):
            return 0
        text = str(x)
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _to_bool(x) -> bool:
        return str(x).strip().lower() in {"true", "1", "yes", "verified purchase"}

    @staticmethod
    def _strip_noise(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _reduce_repetitions(text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1", text)

    def _normalize_pidgin(self, text: str) -> str:
        tokens = text.split()
        normalized = [self.PIDGIN_MAP.get(tok, tok) for tok in tokens]
        return " ".join(normalized)

    @staticmethod
    def _derive_user_id(user_name: str, product_url: str) -> str:
        """
        Since Jumia may not expose a stable reviewer ID, derive a deterministic pseudo ID.
        """
        user_name = str(user_name).strip().lower()
        base = f"{user_name}|{product_url.split('?')[0]}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _compose_text(title: str, body: str) -> str:
        return f"{str(title).strip()} {str(body).strip()}".strip()

    def clean_structure(self) -> pd.DataFrame:
        self.df = self._ensure_columns(self.df)

        self.df["Review_Text"] = self.df["Review_Text"].fillna("").astype(str)
        self.df["Review_Title"] = self.df["Review_Title"].fillna("").astype(str)
        self.df["User_Name"] = self.df["User_Name"].fillna("Anonymous").astype(str)
        self.df["Category"] = self.df["Category"].fillna("Unknown").astype(str)
        self.df["Product_URL"] = self.df["Product_URL"].fillna("").astype(str)

        self.df["Full_Review"] = self.df.apply(
            lambda row: self._compose_text(row["Review_Title"], row["Review_Text"]), axis=1
        )

        before = len(self.df)

        self.df = self.df[self.df["Full_Review"].str.strip() != ""].copy()
        self.df.drop_duplicates(
            subset=["Product_URL", "User_Name", "Review_Text", "Rating", "Timestamp"],
            inplace=True
        )

        self.df["Rating"] = self.df["Rating"].apply(self._parse_rating)
        self.df["Verified_Badge"] = self.df["Verified_Badge"].apply(self._to_bool)
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"], errors="coerce", dayfirst=True)

        self.df["User_ID"] = self.df.apply(
            lambda row: self._derive_user_id(row["User_Name"], row["Product_URL"]), axis=1
        )

        self.df = self.df[self.df["Timestamp"].notna()].copy()

        after = len(self.df)
        print(f"[INFO] Structure cleanup complete. Removed {before - after} rows.")
        return self.df

    def normalize_texts(self) -> pd.DataFrame:
        def normalize(text: str) -> str:
            text = text.lower()
            text = self._strip_noise(text)
            text = self._reduce_repetitions(text)
            text = self._normalize_pidgin(text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        self.df["Cleaned_Text"] = self.df["Full_Review"].apply(normalize)
        self.df["Text_Length"] = self.df["Cleaned_Text"].str.len()
        self.df["Word_Count"] = self.df["Cleaned_Text"].str.split().apply(len)
        return self.df

    def remove_near_duplicates(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        Lightweight near-duplicate removal using normalized exact key.
        If you want heavier semantic duplicate removal, do that in features.py with cosine similarity.
        """
        self.df["Near_Dupe_Key"] = self.df["Cleaned_Text"].str.replace(r"\s+", " ", regex=True)
        before = len(self.df)
        self.df.drop_duplicates(subset=["Near_Dupe_Key"], inplace=True)
        after = len(self.df)
        print(f"[INFO] Removed {before - after} near-duplicate rows.")
        self.df.drop(columns=["Near_Dupe_Key"], inplace=True)
        return self.df

    def engineer_basic_behaviour(self) -> pd.DataFrame:
        product_mean = self.df.groupby("Product_URL")["Rating"].transform("mean")
        self.df["Product_Avg_Rating"] = product_mean.round(3)
        self.df["Rating_Deviation"] = (self.df["Rating"] - self.df["Product_Avg_Rating"]).abs()

        self.df["Review_Date"] = self.df["Timestamp"].dt.date.astype(str)
        product_day_counts = self.df.groupby(["Product_URL", "Review_Date"])["Review_Text"].transform("count")
        self.df["Reviews_Per_Product_Day"] = product_day_counts

        product_day_mean = self.df.groupby("Product_URL")["Reviews_Per_Product_Day"].transform("mean")
        product_day_std = self.df.groupby("Product_URL")["Reviews_Per_Product_Day"].transform("std").fillna(0)

        self.df["Review_Burstiness_Score"] = np.where(
            product_day_std == 0,
            0,
            (self.df["Reviews_Per_Product_Day"] - product_day_mean) / (product_day_std + 1e-6)
        )

        user_review_count = self.df.groupby("User_ID")["Review_Text"].transform("count")
        self.df["User_Review_Count"] = user_review_count

        return self.df

    def proxy_label_reviews(self, rd_threshold: float = 1.5, burst_threshold: float = 2.0) -> pd.DataFrame:
        """
        Proxy labelling:
        deceptive = high rating deviation + no verified badge OR strong burstiness + no verified badge
        genuine = verified + low deviation
        uncertain rows stay -1 and can be excluded from supervised training
        """
        self.df["Deception_Label"] = -1

        deceptive_mask = (
            (~self.df["Verified_Badge"]) &
            (
                (self.df["Rating_Deviation"] >= rd_threshold) |
                (self.df["Review_Burstiness_Score"] >= burst_threshold)
            )
        )

        genuine_mask = (
            (self.df["Verified_Badge"]) &
            (self.df["Rating_Deviation"] < rd_threshold)
        )

        self.df.loc[deceptive_mask, "Deception_Label"] = 1
        self.df.loc[genuine_mask, "Deception_Label"] = 0

        print(self.df["Deception_Label"].value_counts(dropna=False).to_dict())
        return self.df

    def save_outputs(self) -> None:
        os.makedirs("data/processed", exist_ok=True)

        full_path = "data/processed/cleaned_labeled_reviews.csv"
        train_path = "data/processed/trainable_reviews.csv"

        self.df.to_csv(full_path, index=False)
        self.df[self.df["Deception_Label"].isin([0, 1])].to_csv(train_path, index=False)

        meta = {
            "total_rows": int(len(self.df)),
            "trainable_rows": int(self.df["Deception_Label"].isin([0, 1]).sum()),
            "columns": list(self.df.columns)
        }
        with open("data/processed/preprocessing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[SUCCESS] Saved full dataset to {full_path}")
        print(f"[SUCCESS] Saved train-ready dataset to {train_path}")


if __name__ == "__main__":
    raw_files = os.getenv("FILES")
    if not raw_files:
        raise ValueError("FILES variable not found in .env")

    files = [f.strip() for f in raw_files.split(",") if f.strip()]

    pre = JumiaPreprocessor(files)
    pre.load_and_merge()
    pre.clean_structure()
    pre.normalize_texts()
    pre.remove_near_duplicates()
    pre.engineer_basic_behaviour()
    pre.proxy_label_reviews()
    pre.save_outputs()