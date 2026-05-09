# 🎙️ Interrogative Utterance Classification in Malayalam

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-Malayalam-success?style=for-the-badge)
![Speech Processing](https://img.shields.io/badge/Speech-Processing-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Project-purple?style=for-the-badge)

</p>

---

# 🧠 Overview

This project focuses on the **automatic classification of interrogative and non-interrogative Malayalam utterances** using machine learning techniques.

The system investigates how different feature representations contribute to interrogative sentence classification in Malayalam speech and text.

Three **Logistic Regression-based classification models** are developed and compared:

1. **Textual Feature Model**
2. **Acoustic Feature Model**
3. **Hybrid Feature Model** *(Textual + Acoustic)*

This work contributes to research in:

- Malayalam Natural Language Processing (NLP)
- Malayalam Speech Processing
- Computational Linguistics
- Low-resource Language Technologies

---

# 🎯 Objectives

- Develop an automatic interrogative utterance classification system for Malayalam
- Extract and analyze textual linguistic features
- Extract acoustic speech features from Malayalam utterances
- Integrate textual and acoustic features into a hybrid model
- Compare the performance of different feature modalities

---

# 📂 Project Structure

```text
interrogative-classification-malayalam/
│
├── data/
│   ├── speech/
│   │
│   └── text/
│       ├── raw/
│       │   ├── story_1.txt
│       │   └── story_2.txt
│       │
│       └── processed/
│           ├── text_sentences1.csv
│           └── train_sentences1.csv
│
├── src/
│   ├── feature_extraction/
│   │   ├── acoustic_features.py
│   │   ├── textual_features.py
│   │   └── hybrid_features.py
│   │
│   ├── modeling/
│   │   ├── acoustic_model/
│   │   │   ├── data/
│   │   │   │   ├── speech_features_train.json
│   │   │   │   └── speech_features_test.json
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   │
│   │   ├── textual_model/
│   │   │   ├── data/
│   │   │   │   ├── train_dataset1.json
│   │   │   │   └── test_prediction1.json
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── predict.py
│   │   │
│   │   └── hybrid_model/
│   │       ├── data/
│   │       │   ├── train_data_text_speech.json
│   │       │   └── test_predictions_text_speech.json
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       └── predict.py
│   │
│   └── trained_models/
│       ├── acoustic_lr.pkl
│       ├── textual_lr.pkl
│       └── hybrid_lr.pkl
│
├── outputs/
│   ├── acoustic_results.txt
│   ├── textual_results.txt
│   └── hybrid_results.txt
│
├── LICENSE
└── README.md
```

---

# 🔹 Model 1 — Textual Feature Classification

This model uses **linguistic and textual features** extracted from Malayalam utterances.

 ✨ Features

- Interrogative markers
- Lexical patterns
- Sentence structure patterns
- Regex-based linguistic patterns

 🤖 Model

- Logistic Regression

---

# 🔹 Model 2 — Acoustic Feature Classification

This model classifies utterances using **speech-based acoustic cues**.

 ✨ Features

- Pitch
- Energy
- Duration
- Prosodic variations

 🤖 Model

- Logistic Regression

---

# 🔹 Model 3 — Hybrid Feature Classification

This model integrates:

- Textual features
- Acoustic speech features

The hybrid model improves classification performance by combining **linguistic** and **prosodic** information.

 🤖 Model

- Logistic Regression

---

# 📊 Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

 📈 Performance Comparison

| Model   | Feature Type        | Accuracy | Precision | Recall | F1-score |
|----------|--------------------|-----------|------------|--------|-----------|
| Model 1 | Textual Features    | 75%       | 78%        | 72%    | 75%       |
| Model 2 | Acoustic Features   | 75%       | 80%        | 68%    | 73%       |
| Model 3 | Hybrid Features     | 83%       | 87%        | 80%    | 83%       |

 🔍 Key Findings

- The **Hybrid Feature Model** achieved the best overall performance.
- Acoustic features showed strong precision but lower recall.
- Combining textual and acoustic information significantly improved classification robustness.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Scikit-learn | Machine learning |
| Pandas | Data processing |
| Parselmouth / Praat | Acoustic feature extraction |
| Regular Expressions (`re`) | Linguistic pattern extraction |
| Joblib | Model serialization |

---

# 📈 Outputs

The system generates:

- Classification reports
- Confusion matrices
- Prediction outputs

Generated files are stored in:

```text
outputs/
```

---

# ⚠️ Dataset Note

The Malayalam speech and text corpus used in this project is not publicly included due to dataset and licensing considerations.

---

# 📖 Research Contribution

This project contributes to the following research areas:

- Malayalam NLP
- Malayalam Speech Processing
- Computational Linguistics
- Interrogative Pattern Recognition
- Low-resource Language Technologies

---

# 🔮 Future Work

- Deep learning-based classification
- Transformer-based speech-text integration
- Larger Malayalam speech corpus
- Real-time interrogative detection system
- Multi-class sentence classification

---

# 👨‍💻 Author

**Ajin Jacob**  
Computational Linguistics • NLP • Speech Processing

---

# 📄 License

This project is licensed under the **MIT License**.
