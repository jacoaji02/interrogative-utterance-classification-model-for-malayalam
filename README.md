# 🎙️ Interrogative Utterance Classification in Malayalam

## 🧠 Overview

This project focuses on the automatic classification of interrogative and non-interrogative Malayalam utterances using machine learning techniques.

The system investigates how different feature representations contribute to interrogative sentence classification in Malayalam speech and text.

Three Logistic Regression-based classification models are developed and compared:

1. Textual Feature Model
2. Acoustic Feature Model
3. Hybrid Feature Model (Textual + Acoustic)

The project contributes to Malayalam Natural Language Processing (NLP) and Speech Processing research, particularly for low-resource Indian languages.

---

# 🎯 Objectives

* Develop an automatic interrogative utterance classification system for Malayalam
* Extract and analyze textual linguistic features
* Extract acoustic speech features from Malayalam utterances
* Integrate textual and acoustic features into a hybrid model
* Compare the performance of different feature modalities

---

# 📂 Project Structure

```text
interrogative-classification-malayalam/
│
├── data/
|   ├── speech/ 
|   ├── text/
│       ├── raw/
|           ├── story_1.txt
|           ├── story_2.txt 
│       ├── processed/
|           ├── text_sentences1.csv
|           ├── train_sentences1.csv
│
├── src/
│   ├── feature_extraction/
│   │   ├── acoustic_features.py
│   │   ├── textual_features.py
│   │   ├── hybrid_features.py
│   │
│   ├── modeling/
│   │   ├── acoustic_model/
|   |   |   ├── data/
|
│   │   ├── acoustic_features.py
│   │   ├── hybrid_features.py
│   │
│   ├── modeling/
│   │   ├── textual_model/
│   │   ├── acoustic_model/
│   │   ├── hybrid_model/
│
├── trained_models/
│   ├── textual_lr.joblib
│   ├── acoustic_lr.joblib
│   ├── hybrid_lr.joblib
│
├── outputs/
│   ├── plots/
│   ├── reports/
│   ├── confusion_matrices/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🔹 Model 1: Textual Feature Classification

This model uses linguistic and textual features extracted from Malayalam utterances.

## Features

* Interrogative markers
* Lexical patterns
* Sentence structure patterns
* Token-based features
* Regex-based linguistic patterns

## Model

* Logistic Regression

---

# 🔹 Model 2: Acoustic Feature Classification

This model classifies utterances using speech-based acoustic cues.

## Features

* Pitch
* Energy
* Duration
* Prosodic variations

## Model

* Logistic Regression

---

# 🔹 Model 3: Hybrid Feature Classification

This model integrates:

* textual features
* acoustic speech features

The hybrid model aims to improve classification accuracy by combining linguistic and prosodic information.

## Model

* Logistic Regression

---

# 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

## Example Performance Table

| Model   | Feature Type      | Accuracy | Precision | Recall | F1-score |
| ------- | ----------------- | -------- | --------- | ------ | -------- |
| Model 1 | Textual Features  | --       | --        | --     | --       |
| Model 2 | Acoustic Features | --       | --        | --     | --       |
| Model 3 | Hybrid Features   | --       | --        | --     | --       |

---

# 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* praatmouth
* Regular Expressions (`re`)
* Joblib
* Matplotlib

---

# 🚀 How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run textual feature model

```bash
python src/modeling/textual_model/train.py
```

---

## 3. Run acoustic feature model

```bash
python src/modeling/acoustic_model/train.py
```

---

## 4. Run hybrid feature model

```bash
python src/modeling/hybrid_model/train.py
```

---

# 📈 Outputs

The system generates:

* classification reports
* confusion matrices
* prediction outputs

Generated outputs are stored in:

```text
outputs/
```

---

# ⚠️ Dataset Note

The Malayalam speech and text corpus used in this project is not publicly included due to dataset and licensing considerations.

---

# 📖 Research Contribution

This project contributes to:

* Malayalam NLP
* Malayalam Speech Processing
* Computational Linguistics
* Interrogative Pattern Recognition
* Low-resource Language Technologies

---

# 🔮 Future Work

* Deep learning-based classification
* Transformer-based speech-text integration
* Larger Malayalam speech corpus
* Real-time interrogative detection system
* Multi-class sentence classification

---

# 📬 Contact

**Author:** Ajin Jacob
**Field:** Computational Linguistics / NLP / Speech Processing

---

# 📄 License

This project is licensed under the MIT License.
