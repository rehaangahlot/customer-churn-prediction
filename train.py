"""
Training Pipeline for Customer Churn Prediction (ANN).
Faithfully reproduces the Codebasics deep learning workflow.
Trains the Artificial Neural Network and exports model artifacts.
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import prepare_training_data, FEATURE_COLUMNS, COLS_TO_SCALE

DATA_URL = "https://raw.githubusercontent.com/codebasics/deep-learning-keras-tf-tutorial/master/11_chrun_prediction/customer_churn.csv"
DATA_PATH = os.path.join("data", "customer_churn.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
REPORTS_DIR = "reports"


def ensure_dataset():
    """Ensure customer_churn.csv exists, downloading if necessary."""
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Downloading from GitHub...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")
    else:
        print(f"Dataset found at {DATA_PATH}.")


def build_ann_model(input_dim=26):
    """
    Constructs the ANN model matching the reference tutorial:
    - Input/Dense layer: 26 units, ReLU
    - Hidden layer: 15 units, ReLU
    - Output layer: 1 unit, Sigmoid
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(26, activation='relu', name='dense_input'),
        layers.Dense(15, activation='relu', name='dense_hidden'),
        layers.Dense(1, activation='sigmoid', name='dense_output')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(epochs=100, batch_size=32):
    """Executes the full training and evaluation workflow."""
    ensure_dataset()

    print("\n--- 1. Loading and Preprocessing Data ---")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw dataset shape: {df.shape}")

    X, y, scaler, feature_names = prepare_training_data(df)
    print(f"Preprocessed features shape: {X.shape}, target shape: {y.shape}")

    # Train / Test split matching the notebook (80/20, random_state=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("\n--- 2. Building & Training ANN Model ---")
    model = build_ann_model(input_dim=X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    print("\n--- 3. Evaluating Model on Test Set ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Generate predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Calculate metrics
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Metrics Summary ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    print(f"                Predicted No Churn    Predicted Churn")
    print(f"Actual No Churn:      {cm[0][0]:<17} {cm[0][1]}")
    print(f"Actual Churn:         {cm[1][0]:<17} {cm[1][1]}")

    print("\n--- Classification Report ---")
    clf_report = classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"])
    print(clf_report)

    # Save artifacts
    print("\n--- 4. Saving Artifacts ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    print(f"Saved ANN model to: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved MinMaxScaler to: {SCALER_PATH}")

    metadata = {
        "model_architecture": "ANN (26 -> 15 -> 1)",
        "features": feature_names,
        "scaled_columns": COLS_TO_SCALE,
        "metrics": {
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {METADATA_PATH}")

    # Generate and save confusion matrix plot if matplotlib/seaborn available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Customer Churn Prediction - Confusion Matrix')
        cm_plot_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix plot to: {cm_plot_path}")
    except Exception as e:
        print(f"Note: Could not save plot: {e}")

    print("\nTraining completed successfully!")
    return model, scaler, metadata


if __name__ == '__main__':
    train_and_evaluate(epochs=100)
