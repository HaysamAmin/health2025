import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ==========================================
# Load ENV variables (defined in .env)
# ==========================================
ACCURACY_TARGET = float(os.getenv("ML_CLASSIFIER_ACCURACY_TARGET", 0.85))
TFIDF_MAX_FEATURES = int(os.getenv("ML_CLASSIFIER_TFIDF_MAX_FEATURES", 5000))
EXPERIMENT_NAME = os.getenv("ML_CLASSIFIER_EXPERIMENT_NAME", "msg_class_v1")
MODEL_VERSION = os.getenv("CLASSIFIER_MODEL_VERSION", "1.0.0")

# ==========================================
# Paths
# ==========================================
DATA_PATH = "data/classification_dataset.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_PATH = f"{MODEL_OUTPUT_DIR}/message_classifier.pkl"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load dataset
# ==========================================
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# Build pipeline
# ==========================================
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        max_features=TFIDF_MAX_FEATURES,
        max_df=0.95,
        min_df=2
    )),
    ("classifier", LinearSVC())
])


print(f"[ML OPS] Starting training: experiment={EXPERIMENT_NAME}, version={MODEL_VERSION}")
print(f"[ML OPS] Using hyperparameters: TFIDF_MAX_FEATURES={TFIDF_MAX_FEATURES}")

pipeline.fit(X_train, y_train)

# ==========================================
# Evaluate
# ==========================================
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"[ML OPS] Accuracy: {accuracy:.4f}")

# Validate (important for ML OPS)
if accuracy < ACCURACY_TARGET:
    raise ValueError(
        f"[ML OPS] Model accuracy {accuracy:.4f} is below required target {ACCURACY_TARGET}"
    )

# ==========================================
# Save model
# ==========================================
joblib.dump(pipeline, MODEL_OUTPUT_PATH)

print(f"[ML OPS] Model saved to {MODEL_OUTPUT_PATH}")
print(f"[ML OPS] Training complete for experiment {EXPERIMENT_NAME} v{MODEL_VERSION}")
