import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

# ========== Download NLTK resources (chỉ tải nếu thiếu) ==========
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# ========== Hàm xử lý text ==========
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

# ========== Main pipeline ==========
def main():
    raw_path = Path("../data/raw/IMDB Dataset.csv")
    processed_path = Path("../data/processed/IMDB_Processed.csv")

    print("Loading data...")
    df = pd.read_csv(raw_path)

    print("Removing duplicates...")
    df = df.drop_duplicates("review")

    print("Processing reviews...")
    df['review'] = df['review'].apply(process)

    print("Removing duplicates after processing...")
    df = df.drop_duplicates("review")

    print("Mapping sentiment...")
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    print("Saving processed file...")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    print("Done! File saved at:", processed_path)


if __name__ == "__main__":
    main()