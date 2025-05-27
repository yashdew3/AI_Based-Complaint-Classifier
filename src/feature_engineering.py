# src/feature_engineering

import pandas as pd
import numpy as np
import nltk
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from data_preparation import clean_text

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra features like ticket length and sentiment score."""
    df["ticket_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    df["sentiment_polarity"] = df["ticket_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

def build_feature_matrix(df: pd.DataFrame):
    """
    Builds and saves the TF-IDF vectorizer and scaler used for training.
    Returns the combined feature matrix and target labels.
    """
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_features = tfidf_vectorizer.fit_transform(df["clean_text"])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Save vectorizer
    joblib.dump(tfidf_vectorizer, "../models/tfidf_vectorizer.joblib")

    # Additional features
    df = add_additional_features(df)
    scaler = MinMaxScaler()
    additional_feats_scaled = scaler.fit_transform(df[["ticket_length", "sentiment_polarity"]])
    additional_df = pd.DataFrame(additional_feats_scaled, columns=["ticket_length", "sentiment_polarity"])

    # Save scaler
    joblib.dump(scaler, "../models/scaler.joblib")

    # Combine features
    final_features = pd.concat([tfidf_df, additional_df], axis=1)

    return final_features, df["issue_type"], df["urgency_level"]

if __name__ == "__main__":
    raw_data_path = "../data/ai_dev_assignment_tickets_complex_1000.xlsx"  # 🔁 update this if needed
    cleaned_data_path = "../data/cleaned_tickets.xlsx"

    if not os.path.exists("../data"):
        os.makedirs("../data")

    # Load raw data
    df = pd.read_excel(raw_data_path)

    # Clean text
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    # Save cleaned data for reusability
    df.to_excel(cleaned_data_path, index=False)
    print(f"✅ Cleaned data saved to: {cleaned_data_path}")

    # Build features
    features, issue_labels, urgency_labels = build_feature_matrix(df)

    print("✅ Feature matrix built successfully!")
    print("Final feature shape:", features.shape)
    print("Sample feature preview:\n", features.head())
