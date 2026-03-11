import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

from src.inference import HybridReviewInference

st.set_page_config(
    page_title="Jumia Review Intelligence Dashboard",
    layout="wide"
)

st.title("Jumia Review Intelligence Dashboard")
st.caption("Hybrid AfriBERTa + Behavioural Analytics for Deceptive Review Detection")

DATA_PATH = "data/processed/cleaned_labeled_reviews.csv"
METRICS_PATH = "data/models/metrics.json"

if not os.path.exists(DATA_PATH):
    st.error("Processed dataset not found. Run preprocessor first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Model Evaluation",
    "Live Prediction"
])

with tab1:
    st.subheader("Dataset Overview")

    total_reviews = len(df)
    verified_reviews = int(df["Verified_Badge"].sum()) if "Verified_Badge" in df.columns else 0
    deceptive_reviews = int((df["Deception_Label"] == 1).sum()) if "Deception_Label" in df.columns else 0
    genuine_reviews = int((df["Deception_Label"] == 0).sum()) if "Deception_Label" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total_reviews)
    c2.metric("Verified Reviews", verified_reviews)
    c3.metric("Proxy Deceptive", deceptive_reviews)
    c4.metric("Proxy Genuine", genuine_reviews)

    st.subheader("Category Distribution")
    if "Category" in df.columns:
        fig_cat = px.histogram(df, x="Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Rating Distribution")
    if "Rating" in df.columns:
        fig_rating = px.histogram(df, x="Rating", nbins=5)
        st.plotly_chart(fig_rating, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Verified vs Non-Verified")
        if "Verified_Badge" in df.columns:
            verified_counts = df["Verified_Badge"].value_counts().reset_index()
            verified_counts.columns = ["Verified_Badge", "Count"]
            fig_verified = px.pie(
                verified_counts,
                names="Verified_Badge",
                values="Count"
            )
            st.plotly_chart(fig_verified, use_container_width=True)

    with col_right:
        st.subheader("Deception Label Distribution")
        if "Deception_Label" in df.columns:
            label_counts = df["Deception_Label"].value_counts().reset_index()
            label_counts.columns = ["Deception_Label", "Count"]
            fig_label = px.bar(
                label_counts,
                x="Deception_Label",
                y="Count"
            )
            st.plotly_chart(fig_label, use_container_width=True)

    if "Rating_Deviation" in df.columns and "Category" in df.columns:
        st.subheader("Rating Deviation by Category")
        fig_dev = px.box(
            df,
            x="Category",
            y="Rating_Deviation",
            color="Category"
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    if "Review_Burstiness_Score" in df.columns:
        st.subheader("Review Burstiness Score Distribution")
        fig_burst = px.histogram(
            df,
            x="Review_Burstiness_Score",
            nbins=40
        )
        st.plotly_chart(fig_burst, use_container_width=True)

    st.subheader("Preview of Processed Reviews")
    preferred_cols = [
        "Category",
        "Product_Name",
        "User_Name",
        "Rating",
        "Verified_Badge",
        "Rating_Deviation",
        "Review_Burstiness_Score",
        "Deception_Label",
        "Cleaned_Text"
    ]
    available_cols = [col for col in preferred_cols if col in df.columns]
    st.dataframe(df[available_cols].head(200), use_container_width=True)


with tab2:
    st.subheader("Model Evaluation")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        m4.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")

        st.subheader("Confusion Matrix")
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Genuine", "Actual Deceptive"],
                columns=["Pred Genuine", "Pred Deceptive"]
            )
            st.dataframe(cm_df, use_container_width=True)

            cm_melt = cm_df.reset_index().melt(id_vars="index")
            cm_melt.columns = ["Actual", "Predicted", "Count"]

            fig_cm = px.density_heatmap(
                cm_melt,
                x="Predicted",
                y="Actual",
                z="Count",
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Classification Report")
        if "report" in metrics:
            report_df = pd.DataFrame(metrics["report"]).transpose()
            st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("Metrics file not found. Run trainer first.")


with tab3:
    st.subheader("Live Review Prediction")

    if not (
        os.path.exists("data/models/text_model.pkl") and
        os.path.exists("data/models/behaviour_model.pkl") and
        os.path.exists("data/models/scaler.pkl")
    ):
        st.warning("Trained model files not found. Run trainer first.")
    else:
        review_text = st.text_area(
            "Enter cleaned or normal review text",
            height=180,
            placeholder="Example: this product is good but stopped working after one week"
        )

        rating = st.slider("Review Rating", min_value=1, max_value=5, value=5)
        product_avg_rating = st.slider("Product Average Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        review_burstiness_score = st.slider("Review Burstiness Score", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        user_review_count = st.number_input("User Review Count", min_value=1, value=1, step=1)

        if st.button("Predict Review Authenticity"):
            if not review_text.strip():
                st.error("Please enter a review.")
            else:
                with st.spinner("Running hybrid inference..."):
                    predictor = HybridReviewInference()
                    result = predictor.predict(
                        cleaned_text=review_text.strip().lower(),
                        rating=float(rating),
                        product_avg_rating=float(product_avg_rating),
                        review_burstiness_score=float(review_burstiness_score),
                        user_review_count=float(user_review_count)
                    )

                if result["label"] == "Deceptive":
                    st.error(f"Prediction: {result['label']}")
                else:
                    st.success(f"Prediction: {result['label']}")

                a1, a2, a3 = st.columns(3)
                a1.metric("Final Probability", result["final_probability"])
                a2.metric("Text Probability", result["text_probability"])
                a3.metric("Behaviour Probability", result["behaviour_probability"])

                st.subheader("Feature Breakdown")
                feature_df = pd.DataFrame({
                    "Feature": [
                        "Text Sentiment Score",
                        "Scaled Rating",
                        "Sentiment Rating Divergence",
                        "Rating Deviation",
                        "Text Length",
                        "Word Count"
                    ],
                    "Value": [
                        result["text_sentiment_score"],
                        result["scaled_rating"],
                        result["sentiment_rating_divergence"],
                        result["rating_deviation"],
                        result["text_length"],
                        result["word_count"]
                    ]
                })
                st.dataframe(feature_df, use_container_width=True)