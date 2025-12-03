# SymptomSphere

## Overview  
SymptomSphere is an educational medical training platform where students interact with virtual patients.  
The system uses:

- **OpenAI NLU** to map natural-language questions to DDXPlus evidence heads (E_**)   
- **OpenAI NLG** to generate grounded first-person patient responses  
- **A session-based reasoning engine** that reveals evidence turn-by-turn  
- **A custom ON_TOPIC/OFF_TOPIC classifier** to ignore smallTalk and redirect the student  

This project includes a complete **ML Ops pipeline**, a **message classification model**, and proper **environment-variable management** for secure configuration.

---

## Project Structure  

```
SymptomSphere/
│
├── apps/
│   ├── api/
│   │   ├── main.py
│   │   ├── models/schema.py
│   │   ├── routers/
│   │   │   ├── patient.py
│   │   │   └── professor.py
│   │   └── domain/
│   │       ├── nlu_openai.py
│   │       ├── nlg_openai.py
│   │       ├── scoring.py
│   │       ├── store.py
│   │       └── codebook.py
│
├── ml/
│   ├── message_classifier.py
│   ├── generate_classification_file.py
│   ├── train_message_classifier.ipynb
│   └── pipeline/
│       └── train_pipeline.py
│
├── models/
│   └── message_classifier.pkl
│
├── data/
│   ├── classification_dataset.csv
│   └── release_conditions.json
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

##  ON_TOPIC / OFF_TOPIC Classifier

This classifier detects if a student's question is clinical (on_topic) or irrelevant (off_topic).  
It improves the flow by preventing irrelevant messages from triggering diagnostic reasoning.

Classifier uses:
- **TF‑IDF Vectorizer**
- **LinearSVC**
- Saved as `message_classifier.pkl`

Integrated in `apps/api/routers/patient.py`:

```
label = classify(body.text)
if label == "off_topic":
    return {
        "response": "I can help you with medical questions. Could you rephrase your concern?",
        "type": "off_topic"
    }
```

---

##  ML Ops Pipeline

Pipeline script: `ml/pipeline/train_pipeline.py`

This executes data ingestion, preprocessing, training, evaluation, accuracy validation, and model export.

### What it does:
- Loads dataset (`classification_dataset.csv`)
- Reads hyperparameters from `.env`
- Builds TF‑IDF + LinearSVC pipeline
- Trains classifier
- Evaluates accuracy
- Validates against accuracy target
- Saves versioned model into `/models/message_classifier.pkl`

### Run it:
```
python ml/pipeline/train_pipeline.py
```

### `.env` variables used:

```
OPENAI_API_KEY=sk-xxxxxxx

CLASSIFIER_THRESHOLD=0.5
CLASSIFIER_MODEL_VERSION=1.0.0

ML_CLASSIFIER_ACCURACY_TARGET=0.85
ML_CLASSIFIER_EXPERIMENT_NAME=msg_class_v1
ML_CLASSIFIER_TFIDF_MAX_FEATURES=5000
```

These satisfy hyperparameters, accuracy targets, experiment version, and hidden configuration requirements.

---

## Environment Variables  
This project uses `.env` to store sensitive and configurable values.  
`.env` must NOT be committed to GitHub.

```
OPENAI_API_KEY=...
CLASSIFIER_THRESHOLD=-0.3
CLASSIFIER_MODEL_VERSION=1.0.0
ML_CLASSIFIER_ACCURACY_TARGET=0.85
ML_CLASSIFIER_EXPERIMENT_NAME=msg_class_v1
ML_CLASSIFIER_TFIDF_MAX_FEATURES=5000
```

---

## Docker

### Backend build:
```
docker compose up --build
```

Environment variables from `.env` are injected at runtime.

---

##  End-to-End Flow

1. Student asks a question  
2. OFF_TOPIC classifier filters irrelevant messages  
3. ON_TOPIC → processed by NLU  
4. Evidence matched to case  
5. NLG returns grounded patient response  
6. Session updates turn-by-turn  

---

## ✔ Why This Meets Assignment Requirements

- ML Ops pipeline with hyperparameters & accuracy validation  
- Hidden environment variables  
- Versioned ML model  
- Notebook prototype + production-ready pipeline  
- Dockerized backend  
- Real classifier integrated into NLU routing  
- Clean architecture and reproducibility  

---

Project developed by **Miguel** for INFO 8665.
