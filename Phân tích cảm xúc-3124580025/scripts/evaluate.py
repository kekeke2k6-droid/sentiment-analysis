import re
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix

# ========== Text preprocessing giống preprocess.py ==========
lemmatizer = WordNetLemmatizer()

def process(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.replace("n't", " not")
    text = re.sub(r"[^\w\s]", '', text)

    tokens = word_tokenize(text)

    keep_words = {
        'not','no','nor','never','without',
        'but','however','although','though','yet',
        'very','too','so','really','extremely','absolutely',
        'hardly','barely','rarely','seldom'
    }

    stop_words = set(stopwords.words('english')) - keep_words

    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words
    ]

    return " ".join(tokens)

def main():
    print("Loading models...")
    tfidf = joblib.load("../models/tfidf.pkl")
    nb_model = joblib.load("../models/nb_model.pkl")

    df = pd.read_csv("../data/processed/IMDB_Processed.csv")

    X = df["review"]
    y = df["sentiment"]

    X_vec = tfidf.transform(X)
    y_pred = nb_model.predict(X_vec)

    print("\n=== Evaluation on full dataset ===")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # ========== Interactive test ==========
    while True:
        review = input("\nNhập review (gõ 'exit' để thoát): ")
        if review.lower() == 'exit':
            break

        review_processed = process(review)
        vec = tfidf.transform([review_processed])
        pred = nb_model.predict(vec)[0]

        print("Positive" if pred == 1 else "Negative")

if __name__ == "__main__":
    main()