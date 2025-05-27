# src/train_and_save_models

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from feature_engineering import build_feature_matrix
from data_preparation import run_data_preparation_pipeline

# Define directories
DATA_PATH = "..data/ai_dev_assignment_tickets_complex_1000.xlsx"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_model_accuracies(results_dict, task_name):
    """Create bar plot for model accuracies."""
    df_results = pd.DataFrame(results_dict.items(), columns=["Model", "Accuracy"])
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Accuracy", y="Model", data=df_results, palette="viridis")
    plt.title(f"{task_name} - Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/{task_name.lower().replace(' ', '_')}_accuracy_plot.png")
    plt.close()


def train_and_save_models(X, y, task_name, filename_prefix):
    """Train, evaluate, and save models. Highlight and save Random Forest for deployment."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", probability=True)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    accuracy_results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {task_name} - {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[name] = accuracy

        print(f"\n📊 {task_name} - {name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Save all models
        model_path = f"{MODEL_DIR}/{filename_prefix}_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Saved model to: {model_path}")

    # Save only the Random Forest model as default for prediction
    best_model = models["Random Forest"]
    best_path = f"{MODEL_DIR}/{filename_prefix}_best_model.joblib"
    joblib.dump(best_model, best_path)
    print(f"✅ Best model (Random Forest) saved as: {best_path}")

    # Plot performance
    plot_model_accuracies(accuracy_results, task_name)


if __name__ == "__main__":
    print("📥 Running full data preparation pipeline...")
    df_clean = run_data_preparation_pipeline(DATA_PATH)

    print("⚙️ Building feature matrix...")
    X, y_issue, y_urgency = build_feature_matrix(df_clean)

    print("\n🔧 Training models for Issue Type")
    train_and_save_models(X, y_issue, task_name="Issue Type", filename_prefix="issue_type")

    print("\n🔧 Training models for Urgency Level")
    train_and_save_models(X, y_urgency, task_name="Urgency Level", filename_prefix="urgency_level")

    print("\n✅ All models trained and saved successfully!")
