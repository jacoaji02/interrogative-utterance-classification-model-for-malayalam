import json
import joblib

model = joblib.load('textual_lr.pkl')
with open('test_dataset1.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

X_test = [item['features'] for item in test_data]
predictions = model.predict(X_test)

results = []
for i, item in enumerate(test_data):
    result = {
        "sent_id": item["sent_id"],
        "story": item["story"],
        "sentence": item["sentence"],
        "predicted_label": int(predictions[i])
    }
    results.append(result)

with open("test_predictions1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)