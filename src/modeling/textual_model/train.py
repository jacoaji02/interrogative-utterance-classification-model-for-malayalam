import json
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data
with open('train_dataset1.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Prepare features and labels
X_train = [item['features'] for item in train_data]
y_train = [item['label'] for item in train_data]

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'textual_lr.pkl')
print(f"Model trained on {len(X_train)} samples and saved!")
