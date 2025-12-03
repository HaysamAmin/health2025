import json
import csv
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

#############################################
# 1. LOAD ON_TOPIC QUESTIONS FROM PROJECT DATASET
#############################################

def load_on_topic_questions():
    questions = []

    # Dataset 1: release_evidences.json
    if os.path.exists("data/release_evidences.json"):
        with open("data/release_evidences.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, obj in data.items():            
                if "question_en" in obj:
                    q = obj["question_en"].strip()
                    print(f"Loaded question: {q}")
                    if q:
                        questions.append(q)
    return list(set(questions))  # remove duplicates


#############################################
# 2. LOAD OFF_TOPIC FROM KAGGLE DATASET
#############################################

def load_off_topic_questions():


    off_topic = []

    file_path = "Small_talk_Intent.csv"

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "salmanfaroz/small-talk-intent-classification-data",
        file_path,
    )


    if "Utterances" in df.columns:
        texts = df["Utterances"].dropna().astype(str).tolist()
    elif "Sentences" in df.columns:  
        texts = df["Sentences"].dropna().astype(str).tolist()
    else:
        # fallback
            texts = df.iloc[:, 0].dropna().astype(str).tolist()


    
    for t in texts:
        t = t.strip()
        if len(t) > 3:
            off_topic.append(t)

    return off_topic


#############################################
# 3. SAVE TO CSV
#############################################

def save_to_csv(on_topic, off_topic, out_file="data/classification_dataset.csv"):
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for q in on_topic:
            writer.writerow([q, "on_topic"])

        for q in off_topic:
            writer.writerow([q, "off_topic"])

    print(f"Saved dataset → {out_file}")
    print(f"ON_TOPIC: {len(on_topic)} | OFF_TOPIC: {len(off_topic)}")


#############################################
# 4. RUN
#############################################


if __name__ == "__main__":
    on_topic = load_on_topic_questions()
    off_topic = load_off_topic_questions()

    save_to_csv(on_topic, off_topic)
