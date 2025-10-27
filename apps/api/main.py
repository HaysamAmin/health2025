# main.py — FastAPI app factory; loads .env; sets CORS; registers routers;
# creates singletons for Codebook, NLU, and SessionStore.

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from .routers import patient, professor
from .domain.codebook import Codebook      # your robust version
from .domain.nlu_openai import OpenAINLU
from .domain.store import SessionStore

from dotenv import load_dotenv
load_dotenv()
def create_app() -> FastAPI:
    load_dotenv()  # read .env into environment (OPENAI_API_KEY, etc.)
    app = FastAPI(title="SymptSpher API", version="1.0")

    # CORS: allow Streamlit origin to call our API
    allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Singletons: load once, reuse across requests
    app.state.codebook = Codebook("data/release_evidences.json", "data/release_conditions.json")
    app.state.nlu = OpenAINLU()
    app.state.store = SessionStore("cases/sample_cases.jsonl")  # swap to DB later

    # Routers
    app.include_router(patient.router, prefix="/v1/patient", tags=["patient"])
    app.include_router(professor.router, prefix="/v1/professor", tags=["professor"])
    return app

# Uvicorn will import this symbol
app = create_app()

@app.get("/health")
def health():
    import os
    key = os.getenv("OPENAI_API_KEY") or ""
    return {"status":"ok","has_key":bool(key), "prefix": key[:3]}