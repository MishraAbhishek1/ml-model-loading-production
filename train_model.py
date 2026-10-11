from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

texts = [
    "I Love this product",
    "This is an excellent product",
    "Amazing quality and performance",
    "I hate this prodcut",
    "This is a  terriable prodcut",
    "very bad quality"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
]


model = Pipeline([
    ("Vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression()),
])

model.fit(texts, labels)

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")