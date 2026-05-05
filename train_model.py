import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocessing import preprocess

# Load dataset
df = pd.read_csv("product_reviews.csv")

# -------------------------------
# 🧼 DATA CLEANING
# -------------------------------

# Keep only required columns
df = df[['Text', 'Label']]

# Remove null values
df.dropna(subset=['Text', 'Label'], inplace=True)

# Remove empty strings
df = df[df['Text'].str.strip() != ""]

# Normalize labels
df['Label'] = df['Label'].str.strip().str.capitalize()

# Keep only valid labels
df = df[df['Label'].isin(['Positive', 'Negative', 'Neutral'])]

# Remove duplicates
df.drop_duplicates(inplace=True)

# SMART SHORT TEXT FILTER (your chosen logic)
df = df[df['Text'].str.strip().str.len() > 3]
df = df[df['Text'].str.split().apply(len) >= 2]

# Optional advanced cleaning
df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)
df['Text'] = df['Text'].str.replace(r'\d+', '', regex=True)
df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)

# -------------------------------
# TEXT PREPROCESSING
# -------------------------------

df['cleaned'] = df['Text'].apply(preprocess)

# Remove rows where preprocessing gives empty text
df = df[df['cleaned'].str.strip() != ""]

# -------------------------------
# TRAINING
# -------------------------------

X = df['cleaned']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LinearSVC()
model.fit(X_train_vec, y_train)

# -------------------------------
# SAVE MODEL
# -------------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully with cleaned dataset!")