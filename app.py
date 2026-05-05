import streamlit as st
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import preprocess
from utils import get_sentiment_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="sentiPlus",
    layout="wide"
)

# -------------------------------
# BACKGROUND STYLE
# -------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #4facfe, #00f2fe);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown(
    "<h1 style='text-align: center; color: white;'>sentiPlus</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center; color: white;'>Smart Sentiment Analysis Dashboard</h3>",
    unsafe_allow_html=True
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# USER INPUT
# -------------------------------
product = st.text_input("Enter Product Name")
review = st.text_area("Enter Product Review")

# -------------------------------
# ANALYZE
# -------------------------------
if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = preprocess(review)
        vec = vectorizer.transform([cleaned])

        # SVM prediction
        prediction = model.predict(vec)[0]

        # VADER score
        score = get_sentiment_score(review)

        # -------------------------------
        # HYBRID LOGIC
        # -------------------------------
        if score >= 60:
            category = "Highly Positive" if score >= 80 else "Positive"

        elif score <= 40:
            category = "Highly Negative" if score <= 20 else "Negative"

        else:
            # uncertain → SVM
            if prediction == "Positive":
                category = "Positive"
            elif prediction == "Negative":
                category = "Negative"
            else:
                category = "Neutral"

        # -------------------------------
        # RESULTS
        # -------------------------------
        st.subheader("Results")
        st.write(f"**Product:** {product}")
        st.write(f"**Sentiment Quotient:** {score}/100")

        # Color Badge
        if category == "Highly Positive":
            color = "#00C853"
        elif category == "Positive":
            color = "#64DD17"
        elif category == "Neutral":
            color = "#FFD600"
        elif category == "Negative":
            color = "#FF9100"
        else:
            color = "#D50000"

        st.markdown(
            f"""
            <div style="
                display: inline-block;
                padding: 10px 20px;
                border-radius: 25px;
                background-color: {color};
                color: black;
                font-weight: bold;
                font-size: 18px;
            ">
                {category}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------
        # GAUGE BAR
        # -------------------------------
        st.subheader("Sentiment Score Meter")

        fig, ax = plt.subplots()
        ax.barh(["Score"], [score])
        ax.set_xlim(0, 100)
        ax.set_title("Sentiment Quotient")
        st.pyplot(fig)

        # -------------------------------
        # -------------------------------
        # WORD CLOUD
        # -------------------------------
        st.subheader("Word Cloud")

        if review.strip() == "":
            st.warning("Please enter a review to generate word cloud.")
        else:
            try:
                wc = WordCloud(width=800, height=400).generate(review)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc)
                ax2.axis("off")
                st.pyplot(fig2)
            except:
                st.warning("Not enough meaningful words to generate word cloud.")