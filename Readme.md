# 🛠️ AI-Powered Support Ticket Classifier

A complete machine learning pipeline that classifies customer support tickets by issue type and urgency level, and extracts critical entities like product names, complaint keywords, and dates — all accessible through a sleek Gradio web interface.

---

## 📌 Project Overview

Customer support teams handle thousands of tickets daily. Manually classifying and extracting key information from these tickets is time-consuming and error-prone. This project automates:

- 🔍 Ticket Classification: Predicts issue_type and urgency_level.

- 🧠 Entity Extraction: Extracts key entities such as product names, complaint keywords, and dates.

- 🌐 Gradio Interface: Allows end users to interactively classify tickets in a user-friendly UI.

---

## 🚀 Features

- Data Cleaning & Preprocessing (lowercasing, lemmatization, stopwords removal)

- TF-IDF & Feature Engineering (ticket length, sentiment scores)

- Multi-Task Classification using Logistic Regression, SVM, and Random Forest

- Entity Extraction using rule-based NLP (product names, dates, complaint terms)

- Saved Models for inference reuse

- Gradio Frontend for real-time predictions

---

## 📁 Project Structure
```bash
📦 ai_based_complaint_classifier
├── data/
│   ├── ai_dev_assignment_tickets_complex_1000.xlsx
│   └── cleaned_tickets.xlsx
│ 
├── models/
│   ├── issue_type_model.pkl
│   ├── urgency_level_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── feature_scaler.pkl
│ 
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── entity_extraction.py
│   └── predict_ticket.py
│ 
├── app/
│   └── app.py
│ 
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### 1. Clone the repository:
```bash
git clone https://github.com/yashdew3/AI_Based-Complaint-Classifier.git
cd ai_based_complaint_classifier
```

### 2. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Gradio App
```bash
cd src
python app.py
```

---

## 🧠 ML Models

- Multi-class Classification

    - issue_type: Logistic Regression, SVM, Random Forest (best one selected)

    - urgency_level: Trained similarly

- Entity Extraction

    - Regex + Keyword-based patterns for products, dates, and complaint terms

---

## ✨ Sample Input

Input:

    "I received the EcoBreeze AC and it’s broken. Also, I ordered on April 5th."

Output:
```bash
{
  "issue_type": "Product Defect",
  "urgency_level": "High",
  "extracted_entities":
  {
    "products": ["EcoBreeze AC"],
    "dates": ["April 5th"],
    "complaint_keywords": ["broken"]
  }
}
```

---

## 📚 Dependencies

- `pandas`, `numpy`, `nltk`, `scikit-learn`, `textblob`

- `gradio` for UI

---

## 📌 Future Improvements

- Integrate deep learning models (e.g., `BERT`)

- Use Named Entity Recognition (NER) with spaCy or transformers

- Support multilingual ticket classification

- `Database`/`API` backend for enterprise usage

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR for enhancements or fixes. Feel free to check the [issues page](https://github.com/yashdew3/AI_Based-Complaint-Classifier/issues) (if you have one) or open a new issue to discuss changes. Pull requests are also appreciated.

---


## 🧑‍💻 Author

- Built by **Yash Dewangan**
- Github: [YashDewangan](https://github.com/yashdew3)
- Email: [yashdew06@gmail.com](mailto:yashdew06@gmail.com)
- Linkedin: [YashDewangan](https://www.linkedin.com/in/yash-dewangan/)