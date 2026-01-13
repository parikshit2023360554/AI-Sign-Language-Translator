import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    # Setup results dir
    if not os.path.exists('results'):
        os.makedirs('results')

    print("Loading dataset...")
    train_df = pd.read_csv('data/raw_dataset/sign_mnist_train.csv')
    test_df = pd.read_csv('data/raw_dataset/sign_mnist_test.csv')

    y_train = train_df['label'].values
    x_train = train_df.drop('label', axis=1).values
    
    y_test = test_df['label'].values
    x_test = test_df.drop('label', axis=1).values

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)

    # Evaluation
    print("Evaluating model...")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # --- VISUALIZATION & REPORTING ---

    # 1. Save Text Metrics
    report = classification_report(y_test, y_pred)
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("Saved metrics to results/metrics.txt")

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=sorted(list(set(y_test))), yticklabels=sorted(list(set(y_test))))
    plt.title('Confusion Matrix - Sign Language Recognition')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("Saved plot to results/confusion_matrix.png")

    # 3. Accuracy Bar Chart (Per Class approximately)
    # We can parse the report or just plot general accuracy vs random guess
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_accuracies = []
    class_names = []
    
    # Map indices to letters 0->A, 1->B etc.
    # Note: J (9) and Z (25) are not in dataset usually
    labels_map = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }

    for label_str, metrics in report_dict.items():
        if label_str.isdigit():
            class_id = int(label_str)
            class_name = labels_map.get(class_id, str(class_id))
            class_names.append(class_name)
            class_accuracies.append(metrics['f1-score']) # F1 is a balanced metric

    plt.figure(figsize=(14, 6))
    sns.barplot(x=class_names, y=class_accuracies, palette='viridis')
    plt.title('F1-Score per Class (Sign Language Letters)')
    plt.xlabel('Letter')
    plt.ylabel('F1 Score (0-1)')
    plt.ylim(0, 1.0)
    plt.savefig('results/accuracy_per_class.png')
    plt.close()
    print("Saved plot to results/accuracy_per_class.png")

    # Save Model
    with open('models/pixel_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved.")

if __name__ == "__main__":
    train_and_evaluate()
