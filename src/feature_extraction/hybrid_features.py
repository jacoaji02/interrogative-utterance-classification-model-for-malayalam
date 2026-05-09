import pandas as pd
import re
import json
from parselmouth.praat import call
from pathlib import Path

train_df = pd.read_csv("train_sentences1.csv")
file_path = Path("D:/MACL/MA semester 4/dissertaion/textual processing/part_2/train_speech")

def normalize_id(id_str):
    return id_str.split("_")[1].split(".")[0]

WH_PATTERN   = r'(എന്ത്|എന്തിന്|എവിടെ|എപ്പോൾ|എങ്ങനെ|എത്ര|എന്തുകൊണ്ട്|എന്താണ്|ആര്|ആരെ|ആരുടെ|ആർക്ക്|ആരാണ്|ഏത്|ഏതാണ്)'
ALLE_PATTERN = r"അല്ലേ[^\w]*$"
ILLE_PATTERN = r"ഇല്ലേ[^\w]*$"
OO_PATTERN   = r"\w+[ോ][^\w]*$"
EE_PATTERN   = r"\w+[ലേ][^\w]*$"

def extract_features_text(sentence):
    s = sentence.strip()

    return {
        "wh_pattern_detection": int(bool(re.search(WH_PATTERN, s))),
        "ends_with_alle": int(bool(re.search(ALLE_PATTERN, s))),
        "ends_with_ille": int(bool(re.search(ILLE_PATTERN, s))),
        "ends_with_oo": int(bool(re.search(OO_PATTERN, s))),
        "ends_with_ee": int(bool(re.search(EE_PATTERN, s)))
    }

feature_rows = train_df["sentence"].apply(extract_features_text)
feature_df_text = pd.DataFrame(list(feature_rows))

text_data = {}

for i in range(len(train_df)):
    file_id = normalize_id(train_df.iloc[i]["Id"])

    text_features = extract_features_text(train_df.iloc[i]["sentence"])

    text_data[file_id] = {
        "sent_id": file_id,
        "story": train_df.iloc[i]["story"],
        "sentence": train_df.iloc[i]["sentence"],
        "label": train_df.iloc[i]["type"],
        "text_features": list(text_features.values())
    }


id_to_type = {
    normalize_id(row["Id"]): int(row["type"])
    for _, row in train_df.iterrows()
}


def extract_features_speech(tg_path):
    objects = call("Read from file", str(tg_path))
    tg = objects[1]
    sound = objects[0]

    num_intervals = call(tg, "Get number of intervals", 1)

    begin_utterance = call(tg, "Get start time of interval", 1, 2)
    end_utterance = call(tg, "Get end time of interval", 1, num_intervals-1)

    begin_lpw = call(tg, "Get start time of interval", 1, num_intervals-1)
    end_lpw = call(tg, "Get end time of interval", 1, num_intervals-1)

    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    intensity = call(sound, "To Intensity", 75, 0.0)


   
    mean_pitch_lpw = call(pitch, "Get mean", begin_lpw, end_lpw, "Hertz")
    max_pitch_time = call(pitch, "Get time of maximum", 0, 0, "Hertz", "Parabolic")
    min_pitch_time = call(pitch, "Get time of minimum", 0, 0, "Hertz", "Parabolic")
    difference_pitch_time = max_pitch_time - min_pitch_time
    max_pitch_time_lpw = call(pitch, "Get time of maximum", begin_lpw, end_lpw, "Hertz", "Parabolic")
    min_pitch_time_lpw = call(pitch, "Get time of minimum", begin_lpw, end_lpw, "Hertz", "Parabolic")
    difference_pitch_time_lpw = max_pitch_time_lpw - min_pitch_time_lpw
    mean_intensity_lpw = call(intensity, "Get mean", begin_lpw, end_lpw, "dB")
    duration_of_lpw = end_lpw - begin_lpw


    return {
        "min_pitch_time": float(round(min_pitch_time, 3)),
        "difference_pitch_time": float(round(difference_pitch_time, 3)),
        "difference_pitch_time_lpw": float(round(difference_pitch_time_lpw, 3)),
        "mean_intensity_lpw": float(round(mean_intensity_lpw, 3)),
        "duration_of_lpw": float(round(duration_of_lpw, 3))
    }


all_data = []

for file in file_path.glob("*.TextGrid"):
    file_id = normalize_id(file.name)


    if file_id in text_data:
        speech_type = id_to_type[file_id]
        speech_features = extract_features_speech(file)

        combined_features = (
            text_data[file_id]["text_features"] +
            [
                speech_features["min_pitch_time"], 
                speech_features["difference_pitch_time"], 
                speech_features["difference_pitch_time_lpw"],
                speech_features["mean_intensity_lpw"], 
                speech_features["duration_of_lpw"]
            ]

        )
    
        record = {
            "sentence_id": text_data[file_id]["sent_id"],
            "story": text_data[file_id]["story"],
            "sentence": text_data[file_id]["sentence"],
            "speech_id": file.name,
            "label": int(speech_type),
            "features": combined_features
        }

        all_data.append(record)

with open("train_data_text_speech.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

    


