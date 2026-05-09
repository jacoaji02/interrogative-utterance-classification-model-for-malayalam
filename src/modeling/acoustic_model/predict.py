import json
import joblib

model = joblib.load('acoustic_lr.pkl')
with open('speech_test_reduced_2.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

X_test = [item["features"] for item in test_data]
predictions = model.predict(X_test)

results = []
for i, item in enumerate(test_data):
    result = {
        "file_name": item["file"],
        "features": item["features"],
        "actual_label": item["label"],
        "predicted_label": int(predictions[i])
    }
    results.append(result)

with open("test_predictions_speech_with_actual_label_2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)