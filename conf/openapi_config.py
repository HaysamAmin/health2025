import os
from pathlib import Path
from dotenv import load_dotenv

# --- Load .env only once from project root ---
ROOT_DIR = Path(__file__).resolve().parents[1] 
ENV_PATH = ROOT_DIR / ".env"

if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f" .env file not found at {ENV_PATH}")

# --- Global environment variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_NLU_MODEL = os.getenv("OPENAI_NLU_MODEL", "gpt-4o-mini")

# --- Optional safety checks ---
if not OPENAI_API_KEY:
    print(" OPENAI_API_KEY not found in environment variables.")
else:
    print(f" OPENAI_API_KEY loaded successfully (prefix: {OPENAI_API_KEY[:3]})")

print(f" Using NLU model: {OPENAI_NLU_MODEL}")
