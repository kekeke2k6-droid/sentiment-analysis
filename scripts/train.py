import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def main():
    print("Loading data...")
    df = pd.read_csv("../data/processed/IMDB_Processed.csv")

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    joblib.dump(tfidf, "../models/tfidf.pkl")

    # ========== Logistic Regression ==========
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train_tfidf, y_train)
    joblib.dump(lr, "../models/lr_model.pkl")

    # ========== Naive Bayes ==========
    print("Training Naive Bayes...")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_tfidf, y_train)
    joblib.dump(nb, "../models/nb_model.pkl")

    # ========== SVM ==========
    print("Training SVM...")
    svm = LinearSVC(C=0.1)
    svm.fit(X_train_tfidf, y_train)
    joblib.dump(svm, "../models/svm_model.pkl")

    print("Done training and saving models!")

if __name__ == "__main__":
    main()