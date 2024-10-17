import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

KEYWORDS = ["food", "drink"]


def download_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), "../nltk_data")
    if not os.path.exists(nltk_data_dir):
        print("Downloading NLTK data...")
        os.makedirs(nltk_data_dir)
        nltk.download("punkt", download_dir=nltk_data_dir)
        nltk.download("punkt_tab", download_dir=nltk_data_dir)
        nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)


def normalize_text(text):
    if not isinstance(text, str):
        return []  # Return an empty list for non-string values
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


def filter_articles_by_keywords(df, keywords):
    keyword_pattern = "|".join(keywords)
    filtered_articles = df[
        df["tags"].str.contains(keyword_pattern, case=False, na=False)
        | df["article_section"].str.contains(keyword_pattern, case=False, na=False)
    ]
    return filtered_articles


def get_word_frequencies(df):
    word_counter = Counter()
    for description in df["description"].fillna("").tolist():
        tokens = normalize_text(description)
        word_counter.update(tokens)
    return word_counter


def create_word_cloud(word_freq):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", max_words=100
    ).generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def main():
    # Download NLTK data
    download_nltk_data()

    # Load the dataset
    df = pd.read_csv("../data.csv")

    # Filter articles by keywords
    filtered_articles = filter_articles_by_keywords(df, KEYWORDS)

    # Calculate word frequencies for filtered articles
    word_freq = get_word_frequencies(filtered_articles)

    # Create a word cloud from the word frequencies
    create_word_cloud(word_freq)

    # Print the most common words
    print("Most common words:")
    for word, freq in word_freq.most_common(50):
        print(f"{word}: {freq}")


if __name__ == "__main__":
    main()
