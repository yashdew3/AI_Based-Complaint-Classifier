# src/entity_extraction.py

import re
import dateparser
import pandas as pd
from typing import List, Dict


# Load complaint keywords
COMPLAINT_KEYWORDS = [
    "broken", "not working", "error", "late", "delayed", "malfunction",
    "stopped", "damaged", "issue", "problem", "defect", "doesn’t work"
]

DATE_REGEX = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s\d{1,2}(?:st|nd|rd|th)?,?\s?\d{2,4}?)'

def extract_product_names(text: str, product_list: List[str]) -> List[str]:
    found = []
    for product in product_list:
        if product.lower() in text.lower():
            found.append(product)
    return found

def extract_dates(text: str) -> List[str]:
    raw_dates = re.findall(DATE_REGEX, text, flags=re.IGNORECASE)
    parsed_dates = [dateparser.parse(date).strftime("%Y-%m-%d") 
                    for date in raw_dates if dateparser.parse(date)]
    return list(set(parsed_dates))

def extract_complaint_keywords(text: str) -> List[str]:
    found = []
    for kw in COMPLAINT_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text, flags=re.IGNORECASE):
            found.append(kw)
    return found

def extract_entities_from_ticket(ticket_text: str, product_list: List[str]) -> Dict:
    return {
        "products": extract_product_names(ticket_text, product_list),
        "dates": extract_dates(ticket_text),
        "complaint_keywords": extract_complaint_keywords(ticket_text)
    }

def apply_entity_extraction(df: pd.DataFrame) -> pd.DataFrame:
    product_list = df['product'].dropna().unique().tolist()
    df["extracted_entities"] = df["ticket_text"].apply(
        lambda x: extract_entities_from_ticket(x, product_list)
    )
    return df


if __name__ == "__main__":
    from data_preparation import run_data_preparation_pipeline

    # Load and clean data
    df = run_data_preparation_pipeline("../data/ai_dev_assignment_tickets_complex_1000.xlsx")

    # Extract entities
    df_entities = apply_entity_extraction(df)

    print("✅ Sample entity extraction:\n")
    print(df_entities[["ticket_text", "extracted_entities"]].head(5))
    print("\nTotal tickets processed:", len(df_entities))
    print("Entity extraction completed successfully.")
