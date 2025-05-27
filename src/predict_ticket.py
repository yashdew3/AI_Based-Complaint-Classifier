# src/predict_ticket

import joblib
import numpy as np
from textblob import TextBlob
from data_preparation import clean_text
from entity_extraction import extract_entities_from_ticket

# Load trained models
issue_model = joblib.load("E:/Vijayi/ai_ticket_classifier/models/issue_type_random_forest.joblib")
urgency_model = joblib.load("E:/Vijayi/ai_ticket_classifier/models/urgency_level_random_forest.joblib")

# Load vectorizer and scaler
vectorizer = joblib.load("E:/Vijayi/ai_ticket_classifier/models/tfidf_vectorizer.joblib")
scaler = joblib.load("E:/Vijayi/ai_ticket_classifier/models/scaler.joblib")

def predict_ticket(ticket_text: str, product_list: list) -> dict:
    """
    Predicts issue type and urgency level for a support ticket,
    and extracts key entities from the text.
    """
    # 1. Clean text
    cleaned = clean_text(ticket_text)

    # 2. Vectorize and build features
    tfidf_vector = vectorizer.transform([cleaned]).toarray()
    ticket_length = len(cleaned.split())
    sentiment_score = TextBlob(ticket_text).sentiment.polarity
    scaled_additional = scaler.transform([[ticket_length, sentiment_score]])
    final_features = np.concatenate([tfidf_vector, scaled_additional], axis=1)

    # 3. Predict labels
    issue_pred = issue_model.predict(final_features)[0]
    urgency_pred = urgency_model.predict(final_features)[0]

    # 4. Extract entities
    extracted_entities = extract_entities_from_ticket(ticket_text, product_list)

    return {
        "issue_type": issue_pred,
        "urgency_level": urgency_pred,
        "extracted_entities": extracted_entities
    }

# Test usage
if __name__ == "__main__":
    example_text = "I received the EcoBreeze AC and it’s broken. Also, I ordered on April 5th."
    product_list = ["EcoBreeze AC", "SmartWatch V2", "PhotoSnap Cam", "SoundWave 300"]

    result = predict_ticket(example_text, product_list)
    print("✅ Prediction Result:")
    print(result)
