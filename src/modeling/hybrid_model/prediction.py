import json
import joblib

model = joblib.load('hybrid_lr.pkl')
with open('test_data_text_speech.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

X_test = [item["features"] for item in test_data]
predictions = model.predict(X_test)

results = []
for i, item in enumerate(test_data):
    result = {
        "sentence_id": item["sentence_id"],
        "story": item["story"],
        "file_name": item["speech_id"],
        "features": item["features"],
        "actual_label": item["label"],
        "predicted_label": int(predictions[i])
    }
    results.append(result)

with open("test_predictions_text_speech_with_actual_label.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
