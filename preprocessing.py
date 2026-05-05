import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words or word in ["not", "no"]]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)