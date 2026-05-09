import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

model = joblib.load("hybrid_lr.pkl")

with open("test_data_text_speech.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

x_test = [item["features"] for item in test_data]
y_true = [item["label"] for item in test_data]

predic = model.predict(x_test)
accuracy = accuracy_score(y_true, predic)
precision = precision_score(y_true, predic)
recall = recall_score(y_true, predic)
f1 = f1_score(y_true, predic)
classification = classification_report(y_true, predic)
confusion = confusion_matrix(y_true, predic)

with open("evaluation_sklearn_text_speech2.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy = {accuracy}\n")
    f.write(f"Precision = {precision}\n")
    f.write(f"Recall= {recall}\n")
    f.write(f"F1 Score = {f1}\n")
    f.write(f"Classification Report: {classification}")
    f.write(f"Confusion Matrix: {confusion}")
