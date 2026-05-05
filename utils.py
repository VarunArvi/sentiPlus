from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    score = sia.polarity_scores(text)['compound']
    quotient = int((score + 1) * 50)
    return quotient

def get_category(score):
    if score >= 80:
        return "Highly Positive"
    elif score >= 60:
        return "Positive"
    elif score >= 40:
        return "Neutral"
    elif score >= 20:
        return "Negative"
    else:
        return "Highly Negative"