# sentiPlus

Hybrid Sentiment Analysis System using SVM + VADER.

## Features
- Sentiment classification (Positive/Negative/Neutral)
- Sentiment Quotient (0–100)
- Hybrid decision logic
- Word Cloud visualization
- Streamlit dashboard

## Architecture
User Input → Preprocessing → (SVM + VADER) → Hybrid Logic → UI

## Tech Stack
Python, scikit-learn, NLTK, Streamlit, Matplotlib, WordCloud

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
