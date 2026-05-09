import parselmouth
import json
from parselmouth.praat import call
from pathlib import Path
import csv
import pandas as pd


file_path = Path("D:/MACL/MA semester 4/dissertaion/textual processing/part_2/train_speech")
train_df = pd.read_csv("train_sentences1.csv")


def normalize_id(id_str):
    return id_str.split("_")[1].split(".")[0]

id_to_type = {
    normalize_id(row["Id"]): row["type"]
    for _, row in train_df.iterrows()
}


def extract_features(tg_path):
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
        "min_pitch_time": round(min_pitch_time, 3),
        "difference_pitch_time": round(difference_pitch_time, 3),
        "difference_pitch_time_lpw": round(difference_pitch_time_lpw, 3),
        "mean_intensity_lpw": round(mean_intensity_lpw, 3),
        "duration_of_lpw": round(duration_of_lpw, 3)
    }


all_data = []
for file in file_path.glob("*.TextGrid"):
    file_id = normalize_id(file.name)


    if file_id in id_to_type:
        speech_type = id_to_type[file_id]
        features = extract_features(file)
        record = {
            "file": file.name,
            "label": int(speech_type),
            "features": [features["min_pitch_time"], features["difference_pitch_time"], features["difference_pitch_time_lpw"],
                         features["mean_intensity_lpw"], features["duration_of_lpw"]]
        }
    
        all_data.append(record)

with open("speech_train_reduced_2.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4)




