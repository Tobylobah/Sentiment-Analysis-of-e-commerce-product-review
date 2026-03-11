import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


class FeatureEngineer:
    def __init__(
        self,
        afriberta_model_name: str = "castorini/afriberta_large"
    ):
        self.afriberta_model_name = afriberta_model_name
        self.tokenizer = None
        self.encoder = None

    def load_encoder(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.afriberta_model_name)
        self.encoder = AutoModel.from_pretrained(self.afriberta_model_name)

    def create_tfidf(self, texts, max_features: int = 5000):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X = vectorizer.fit_transform(texts)
        return vectorizer, X

    def get_afriberta_embeddings(self, texts, batch_size: int = 16):
        if self.tokenizer is None or self.encoder is None:
            self.load_encoder()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(device)
        self.encoder.eval()

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = self.encoder(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    @staticmethod
    def compute_sentiment_rating_divergence(df: pd.DataFrame, text_score_col: str = "Text_Sentiment_Score"):
        """
        rating scale 1..5 -> mapped to [-1, 1]
        """
        df["Scaled_Rating_Sentiment"] = ((df["Rating"] - 1) / 4) * 2 - 1
        df["Sentiment_Rating_Divergence"] = np.abs(
            df[text_score_col] - df["Scaled_Rating_Sentiment"]
        )
        return df

    @staticmethod
    def pairwise_duplicate_similarity(texts):
        matrix = TfidfVectorizer(max_features=3000).fit_transform(texts)
        sim = cosine_similarity(matrix)
        return sim


def save_matrix(path: str, matrix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(matrix, path)