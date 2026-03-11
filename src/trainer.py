import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer, AutoModel, pipeline


class HybridReviewTrainer:
    def __init__(self):
        self.df = pd.read_csv("data/processed/trainable_reviews.csv")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.behaviour_cols = [
            "Rating",
            "Product_Avg_Rating",
            "Rating_Deviation",
            "Review_Burstiness_Score",
            "User_Review_Count",
            "Text_Length",
            "Word_Count",
            "Sentiment_Rating_Divergence"
        ]

    def load_afriberta(self):
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_large")
        self.encoder = AutoModel.from_pretrained("castorini/afriberta_large")
        self.encoder.to(self.device)
        self.encoder.eval()

    def sentiment_signal(self):
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        scores = []
        for text in self.df["Cleaned_Text"].fillna("").tolist():
            try:
                out = sentiment_pipeline(text[:512])[0]
                label = out["label"].lower()
                score = float(out["score"])

                if "negative" in label:
                    scores.append(-score)
                elif "positive" in label:
                    scores.append(score)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)

        self.df["Text_Sentiment_Score"] = scores

    def rating_sentiment_gap(self):
        self.df["Scaled_Rating"] = ((self.df["Rating"] - 1) / 4) * 2 - 1
        self.df["Sentiment_Rating_Divergence"] = np.abs(
            self.df["Text_Sentiment_Score"] - self.df["Scaled_Rating"]
        )

    def text_embeddings(self):
        embeddings = []

        with torch.no_grad():
            for text in self.df["Cleaned_Text"].fillna("").tolist():
                encoded = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.encoder(**encoded)
                cls_vec = output.last_hidden_state[:, 0, :]
                embeddings.append(cls_vec.cpu().numpy()[0])

        return np.array(embeddings)

    def prepare(self):
        self.load_afriberta()
        self.sentiment_signal()
        self.rating_sentiment_gap()

        X_text = self.text_embeddings()
        X_beh = self.df[self.behaviour_cols].fillna(0).values
        y = self.df["Deception_Label"].values

        return X_text, X_beh, y

    def train(self):
        X_text, X_beh, y = self.prepare()

        X_text_train, X_text_test, X_beh_train, X_beh_test, y_train, y_test = train_test_split(
            X_text,
            X_beh,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_beh_train_scaled = scaler.fit_transform(X_beh_train)
        X_beh_test_scaled = scaler.transform(X_beh_test)

        text_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

        behaviour_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        )

        text_model.fit(X_text_train, y_train)
        behaviour_model.fit(X_beh_train_scaled, y_train)

        text_prob = text_model.predict_proba(X_text_test)[:, 1]
        behaviour_prob = behaviour_model.predict_proba(X_beh_test_scaled)[:, 1]

        final_prob = (0.6 * text_prob) + (0.4 * behaviour_prob)
        predictions = (final_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "report": classification_report(y_test, predictions, output_dict=True)
        }

        os.makedirs("data/models", exist_ok=True)

        joblib.dump(text_model, "data/models/text_model.pkl")
        joblib.dump(behaviour_model, "data/models/behaviour_model.pkl")
        joblib.dump(scaler, "data/models/scaler.pkl")

        with open("data/models/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(json.dumps(metrics, indent=4))
        return metrics


if __name__ == "__main__":
    trainer = HybridReviewTrainer()
    trainer.train()