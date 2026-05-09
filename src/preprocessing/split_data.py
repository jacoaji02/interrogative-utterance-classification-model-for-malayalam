import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_labelled_dataset.csv")

train_parts = []
test_parts = []

# loop through each story
for story, story_group in df.groupby("story"):

    # stratified split inside story
    train, test = train_test_split(
        story_group,
        test_size=0.2,
        stratify=story_group["type"],   # ⭐ keeps 1:1 balance
        random_state=42,
        shuffle=True
    )

    train_parts.append(train)
    test_parts.append(test)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

train_df.to_csv("train_sentences1.csv", index=False)
test_df.to_csv("test_sentences1.csv", index=False)