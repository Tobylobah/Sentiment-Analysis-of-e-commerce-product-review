# Final README.md

Use this structure. It matches what your chapters describe and makes your repo defensible academically.

```md
# Hybrid AI System for Detecting Deceptive E-commerce Reviews in Nigeria

## Overview

This project implements a hybrid Artificial Intelligence system for detecting deceptive product reviews on Jumia Nigeria. The system combines deep language understanding using AfriBERTa with behavioural fraud indicators derived from reviewer activity patterns.

The goal is to improve trust in e-commerce platforms by identifying suspicious reviews using both semantic analysis and behavioural analytics.

This system was developed as a final year Computer Science research project.

---

## System Architecture

The system follows a multi-stage machine learning pipeline:

Data Collection Layer  
Jumia review scraping using Selenium and BeautifulSoup.

Data Processing Layer  
Cleaning, normalization, Nigerian Pidgin handling, deduplication.

Feature Engineering Layer  
Text embeddings from AfriBERTa.  
Behaviour features like rating deviation and review burstiness.

Model Layer  
Text classifier (Logistic Regression on AfriBERTa embeddings).  
Behaviour classifier (Random Forest).

Decision Layer  
Late fusion combining both model probabilities.

Evaluation Layer  
Accuracy, Precision, Recall, F1 Score, Confusion Matrix.

Visualization Layer  
Streamlit analytics dashboard.

---

## Project Structure
```

project/

app.py

requirements.txt

.env

data/

raw/

processed/

models/

logs/

src/

scraper.py

preprocessor.py

trainer.py

inference.py

README.md

````

---

## Features Implemented

Review scraping from multiple Jumia categories

Nigerian text normalization

Pidgin-aware preprocessing

Duplicate detection

Behaviour fraud signals:
Rating deviation
Review burstiness
Reviewer activity patterns

AfriBERTa semantic embeddings

Hybrid fraud classification

Late fusion decision system

Model evaluation metrics

Interactive analytics dashboard

Live review prediction interface

---

## Behaviour Features Used

The behavioural fraud indicators include:

Rating Deviation
Difference between review rating and product average.

Review Burstiness
Abnormal number of reviews within short time windows.

Reviewer Activity
Number of reviews per user.

Sentiment Rating Divergence
Difference between text sentiment and rating.

Verified Purchase Indicator
Trust proxy signal.

---

## Installation

Create virtual environment:

```bash
python -m venv venv
````

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create `.env` file:

```
DRIVER=C:/path/to/msedgedriver.exe

AGENT=Mozilla/5.0

FILES=data/raw/jumia_reviews_mobile.csv,data/raw/jumia_reviews_computing.csv,data/raw/jumia_reviews_electronics.csv
```

---

## How To Run The System

Step 1 — Scrape Data

```
python src/scraper.py
```

Step 2 — Preprocess Data

```
python src/preprocessor.py
```

Step 3 — Train Hybrid Model

```
python src/trainer.py
```

Step 4 — Launch Dashboard

```
streamlit run app.py
```

---

## Model Training Strategy

Text Model:

AfriBERTa embeddings extracted from reviews.

Classifier:
Logistic Regression.

Behaviour Model:

Random Forest trained on behavioural fraud indicators.

Fusion Strategy:

Final Probability:

```
Final Score =
0.6 × Text Probability +
0.4 × Behaviour Probability
```

Decision Threshold:

```
>= 0.5 → Deceptive

< 0.5 → Genuine
```

---

## Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Classification Report

---

## Example Prediction

Input:

Review:
"This product is very good but stopped working after one week"

Output:

Label:
Deceptive

Confidence:
0.71

Text Probability:
0.64

Behaviour Probability:
0.79

---

## Limitations

Proxy labels used instead of manually annotated fraud labels.

Dataset limited to selected Jumia categories.

Behaviour signals rely on observable metadata.

Future improvements may include:

Human annotated dataset

Graph fraud detection

Temporal anomaly modelling

Transformer fine-tuning

---

## Future Improvements

Graph neural networks for reviewer networks

Temporal fraud detection models

Semi-supervised learning

Active learning

Fraud explanation models

Mobile deployment

---

## Technologies Used

Python

PyTorch

Transformers

Scikit-learn

Selenium

BeautifulSoup

Streamlit

Plotly

---

## Research Contribution

This project contributes:

Hybrid fraud detection architecture for African e-commerce.

Use of AfriBERTa for review fraud detection.

Behavioural fraud signal engineering.

Practical fraud detection pipeline for low-resource environments.

---

## Author

Computer Science Final Year Project

Hybrid AI Review Intelligence System

Nigeria

```

---

# Final checklist (do these before submission)

Do these and your implementation becomes very strong:

Run full pipeline once:
scraper → preprocessor → trainer → dashboard

Take screenshots of:
Dashboard overview
Confusion matrix
Live prediction tab
Category distribution

Add screenshots to Chapter 4.

Add model results table:

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| Behaviour RF | | | | |
| Text LR | | | | |
| Hybrid | | | | |

Hybrid should perform best (even slightly).



```
