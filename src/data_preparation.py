# src/data_preparation

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def load_data(filepath):
    """Load dataset from Excel file"""
    df = pd.read_excel(filepath)
    return df


def clean_text(text):
    """Normalize and clean ticket text"""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_dataframe(df):
    """Apply text cleaning and handle missing values"""
    df = df.copy()

    # Drop rows where labels are missing
    df.dropna(subset=['issue_type', 'urgency_level'], inplace=True)

    # Fill missing text with empty string
    df['ticket_text'] = df['ticket_text'].fillna("")

    # Apply text cleaning
    df['clean_text'] = df['ticket_text'].apply(clean_text)

    return df


def run_data_preparation_pipeline(filepath):
    df = load_data(filepath)
    cleaned_df = preprocess_dataframe(df)
    return cleaned_df


if __name__ == "__main__":
    # Sample usage
    input_path = "../data/ai_dev_assignment_tickets_complex_1000.xlsx"
    df_cleaned = run_data_preparation_pipeline(input_path)
    print(df_cleaned.head())
