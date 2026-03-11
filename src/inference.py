import joblib
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel, pipeline


class HybridReviewInference:
    """
    Inference engine for the hybrid Jumia review intelligence system.

    Pipeline:
    Text -> AfriBERTa embedding -> text classifier
    Behaviour features -> scaler -> behaviour classifier
    Late fusion -> final probability -> label
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_model = joblib.load("data/models/text_model.pkl")
        self.behaviour_model = joblib.load("data/models/behaviour_model.pkl")
        self.scaler = joblib.load("data/models/scaler.pkl")

        self.tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_large")
        self.encoder = AutoModel.from_pretrained("castorini/afriberta_large")
        self.encoder.to(self.device)
        self.encoder.eval()

        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    def get_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.encoder(**encoded)
            cls_vec = output.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_vec

    def get_text_sentiment_score(self, text: str) -> float:
        try:
            result = self.sentiment_pipe(text[:512])[0]
            label = result["label"].lower()
            score = float(result["score"])

            if "negative" in label:
                return -score
            if "positive" in label:
                return score
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def compute_scaled_rating(rating: float) -> float:
        return ((rating - 1) / 4) * 2 - 1

    def build_behaviour_vector(
        self,
        rating: float,
        product_avg_rating: float,
        rating_deviation: float,
        review_burstiness_score: float,
        user_review_count: float,
        text_length: float,
        word_count: float,
        sentiment_rating_divergence: float
    ) -> np.ndarray:
        return np.array([[
            rating,
            product_avg_rating,
            rating_deviation,
            review_burstiness_score,
            user_review_count,
            text_length,
            word_count,
            sentiment_rating_divergence
        ]])

    def predict(
        self,
        cleaned_text: str,
        rating: float,
        product_avg_rating: float,
        review_burstiness_score: float,
        user_review_count: float
    ) -> dict:
        text_embedding = self.get_text_embedding(cleaned_text)

        text_sentiment_score = self.get_text_sentiment_score(cleaned_text)
        scaled_rating = self.compute_scaled_rating(rating)
        sentiment_rating_divergence = abs(text_sentiment_score - scaled_rating)
        rating_deviation = abs(rating - product_avg_rating)

        text_length = len(cleaned_text)
        word_count = len(cleaned_text.split())

        behaviour_vector = self.build_behaviour_vector(
            rating=rating,
            product_avg_rating=product_avg_rating,
            rating_deviation=rating_deviation,
            review_burstiness_score=review_burstiness_score,
            user_review_count=user_review_count,
            text_length=text_length,
            word_count=word_count,
            sentiment_rating_divergence=sentiment_rating_divergence
        )

        scaled_behaviour = self.scaler.transform(behaviour_vector)

        text_prob = float(self.text_model.predict_proba(text_embedding)[0, 1])
        behaviour_prob = float(self.behaviour_model.predict_proba(scaled_behaviour)[0, 1])

        final_prob = (0.6 * text_prob) + (0.4 * behaviour_prob)
        predicted_label = 1 if final_prob >= 0.5 else 0

        return {
            "label": "Deceptive" if predicted_label == 1 else "Genuine",
            "final_probability": round(final_prob, 4),
            "text_probability": round(text_prob, 4),
            "behaviour_probability": round(behaviour_prob, 4),
            "text_sentiment_score": round(text_sentiment_score, 4),
            "scaled_rating": round(scaled_rating, 4),
            "sentiment_rating_divergence": round(sentiment_rating_divergence, 4),
            "rating_deviation": round(rating_deviation, 4),
            "text_length": int(text_length),
            "word_count": int(word_count)
        }


if __name__ == "__main__":
    model = HybridReviewInference()

    sample_review = "this phone is very good and works perfectly but the charger stopped working after two days"

    result = model.predict(
        cleaned_text=sample_review,
        rating=5,
        product_avg_rating=4.1,
        review_burstiness_score=0.8,
        user_review_count=2
    )

    print(result)