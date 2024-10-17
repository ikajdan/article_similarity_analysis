import os
from time import time

import nltk
import pandas as pd
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
from nltk.tokenize import word_tokenize

# List of related words
RELATED_WORDS_FOOD = ["food", "drink"]
RELATED_WORDS_SPORT = ["sport", "game"]


def download_nltk_data():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        print("Downloading NLTK data...")
        os.makedirs(nltk_data_dir)
        nltk.download("punkt", download_dir=nltk_data_dir)
        nltk.download("stopwords", download_dir=nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)


def normalize_tags(tags):
    if pd.isna(tags):
        return ""
    unique_tags = set(tag.strip().lower() for tag in tags.split(",") if tag.strip())
    return ",".join(unique_tags)


def normalize_text(text):
    if not isinstance(text, str):
        return ""  # Treat non-string values as empty strings
    tokens = word_tokenize(text.lower())
    return tokens


def contains_related_words(text, related_words):
    if not isinstance(text, str):
        return False
    text = text.lower()  # Ensure comparison is case insensitive
    return any(word in text for word in related_words)


def create_corpus(df):
    return [
        normalize_text(tags) + normalize_text(section)
        for tags, section in zip(
            df["tags"].fillna(""),
            df["article_section"].fillna(""),
        )
    ]


def calculate_tfidf(corpus):
    dictionary = corpora.Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf_model = models.TfidfModel(bow_corpus)
    similarity_index = MatrixSimilarity(tfidf_model[bow_corpus])
    return tfidf_model, dictionary, similarity_index


def is_food_related(row, related_words):
    combined_text = f"{row.tags or ''} {row.article_section or ''}"
    return contains_related_words(combined_text.lower(), related_words)


def get_top_similar_articles(tfidf_model, dictionary, similarity_index, article):
    vec_bow = dictionary.doc2bow(normalize_text(article))
    vec_tfidf = tfidf_model[vec_bow]
    sims = similarity_index[vec_tfidf]
    return sorted(enumerate(sims), key=lambda item: -item[1])[:10]


def calculate_quality_ratio(
    df, related_words, tfidf_model, dictionary, similarity_index
):
    num_articles_food_related = df[df["is_food_related"]].shape[0]
    total_goods = 0

    for row in df[df["is_food_related"]].itertuples():
        combined_text = f"{row.tags or ''} {row.article_section or ''}"
        top_10 = get_top_similar_articles(
            tfidf_model, dictionary, similarity_index, combined_text
        )

        goods = sum(
            1
            for index, _ in top_10
            if contains_related_words(
                f"{df.iloc[index].tags or ''} {df.iloc[index].article_section or ''}",
                related_words,
            )
        )
        total_goods += goods

    if num_articles_food_related == 0:
        return 0
    return total_goods / (num_articles_food_related * 10)


def main():
    # Download NLTK data
    download_nltk_data()

    # Load the dataset
    df = pd.read_csv("data.csv")

    # Combine the tags and article section
    df["tags"] = df["tags"].fillna("") + "," + df["article_section"].fillna("")

    # Normalize the tags column
    df["tags"] = df["tags"].apply(normalize_tags)

    # Create a combined corpus from the tags and article section
    corpus = create_corpus(df)

    # Calculate the TF-IDF model, dictionary, and similarity index
    start_time = time()
    tfidf_model, dictionary, similarity_index = calculate_tfidf(corpus)
    print(f"TF-IDF model creation time: {time() - start_time:.4f} seconds")

    # Mark food-related articles
    df["is_food_related"] = df.apply(
        lambda row: is_food_related(row, RELATED_WORDS_FOOD), axis=1
    )

    # Calculate the quality ratio for food-related content
    start_time = time()
    quality_ratio = calculate_quality_ratio(
        df, RELATED_WORDS_FOOD, tfidf_model, dictionary, similarity_index
    )
    print(
        f"Quality ratio for 'Food and Drinks': {quality_ratio:.3f} (in {time() - start_time:.4f} seconds)"
    )


if __name__ == "__main__":
    main()
