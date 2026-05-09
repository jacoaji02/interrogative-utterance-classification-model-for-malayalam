import pandas as pd
import re

df = pd.read_csv('labelled_dataset.csv')

def clean_text(text):
    text = str(text)
    text = re.sub(r"[^\w\s\u0D00-\u0D7F]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["sentence"] = df["sentence"].apply(clean_text) 

df.to_csv('cleaned_labelled_dataset.csv', index=False, encoding="utf-8")
