import os
from time import time

import nltk
import pandas as pd
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
from nltk.tokenize import word_tokenize


def download_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        print("Downloading NLTK data...")
        os.makedirs(nltk_data_dir)
        nltk.download("punkt", download_dir=nltk_data_dir)
        nltk.download("punkt_tab", download_dir=nltk_data_dir)
        nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)


def normalize_text(text):
    if not isinstance(text, str):
        return ""  # Treat non-string values as empty strings
    tokens = word_tokenize(text.lower())
    return tokens


def create_corpus(df):
    return [normalize_text(desc) for desc in df["description"].fillna("").tolist()]


def calculate_tfidf(corpus):
    dictionary = corpora.Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf_model = models.TfidfModel(bow_corpus)
    similarity_index = MatrixSimilarity(tfidf_model[bow_corpus])
    return tfidf_model, dictionary, similarity_index


def is_food_and_drink_section(article_section):
    if not isinstance(article_section, str):
        return False
    return "food & drink" in article_section.lower()


def get_top_similar_articles(tfidf_model, dictionary, similarity_index, article):
    vec_bow = dictionary.doc2bow(normalize_text(article))
    vec_tfidf = tfidf_model[vec_bow]
    sims = similarity_index[vec_tfidf]
    return sorted(enumerate(sims), key=lambda item: -item[1])[:10]


def calculate_quality_ratio(df, tfidf_model, dictionary, similarity_index):
    food_drink_articles = df[df["is_food_drink_section"]]
    num_articles_food_related = food_drink_articles.shape[0]
    total_goods = 0

    for description in food_drink_articles["description"]:
        top_10 = get_top_similar_articles(
            tfidf_model, dictionary, similarity_index, description
        )

        goods = sum(1 for index, _ in top_10 if df.iloc[index]["is_food_drink_section"])
        total_goods += goods

    if num_articles_food_related == 0:
        return 0

    return total_goods / (num_articles_food_related * 10)


def main():
    # Download NLTK data
    download_nltk_data()

    # Load the dataset
    df = pd.read_csv("data.csv")

    # Mark articles that belong to the 'Food & Drink' section
    df["is_food_drink_section"] = df["article_section"].apply(is_food_and_drink_section)

    # Create a corpus from the descriptions
    corpus = create_corpus(df)

    # Calculate the TF-IDF model, dictionary, and similarity index
    start_time = time()
    tfidf_model, dictionary, similarity_index = calculate_tfidf(corpus)
    print(f"TF-IDF model creation time: {time() - start_time:.4f} seconds")

    # Calculate the quality ratio for 'Food & Drink' section articles
    start_time = time()
    quality_ratio = calculate_quality_ratio(
        df, tfidf_model, dictionary, similarity_index
    )
    print(
        f"Quality ratio for 'Food and Drink': {quality_ratio:.3f} (in {time() - start_time:.4f} seconds)"
    )


if __name__ == "__main__":
    main()
