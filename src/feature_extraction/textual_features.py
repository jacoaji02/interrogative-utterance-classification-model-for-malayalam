import pandas as pd
import re
import json

df = pd.read_csv("test_sentences1.csv")

WH_PATTERN   = r'(എന്ത്|എന്തിന്|എവിടെ|എപ്പോൾ|എങ്ങനെ|എത്ര|എന്തുകൊണ്ട്|എന്താണ്|ആര്|ആരെ|ആരുടെ|ആർക്ക്|ആരാണ്|ഏത്|ഏതാണ്)'
ALLE_PATTERN = r"അല്ലേ[^\w]*$"
ILLE_PATTERN = r"ഇല്ലേ[^\w]*$"
OO_PATTERN   = r"\w+[ോ][^\w]*$"
EE_PATTERN   = r"\w+[ലേ][^\w]*$"

def extract_features(sentence):
    s = sentence.strip()

    return {
        "wh_pattern_detection": int(bool(re.search(WH_PATTERN, s))),
        "ends_with_alle": int(bool(re.search(ALLE_PATTERN, s))),
        "ends_with_ille": int(bool(re.search(ILLE_PATTERN, s))),
        "ends_with_oo": int(bool(re.search(OO_PATTERN, s))),
        "ends_with_ee": int(bool(re.search(EE_PATTERN, s)))
    }

feature_rows = df["sentence"].apply(extract_features)
feature_df = pd.DataFrame(list(feature_rows))

records = []

for i in range(len(df)):
    record = {
        "sent_id": df.iloc[i]["Id"],
        "story": df.iloc[i]["story"],
        "sentence": df.iloc[i]["sentence"],
        "label": int(df.iloc[i]["type"]),
        "features": feature_df.iloc[i].to_list()
    }
    records.append(record)

with open("test_dataset1.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)
