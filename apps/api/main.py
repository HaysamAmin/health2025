# main.py — FastAPI app factory; loads .env; sets CORS; registers routers;
# creates singletons for Codebook, NLU, and SessionStore.

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from conf.log_config import logger
from .routers import patient, professor
from .domain.codebook import Codebook      # your robust version
from .domain.nlu_openai import OpenAINLU
from .domain.store import SessionStore
from dotenv import load_dotenv

from conf.openapi_config import OPENAI_API_KEY, OPENAI_NLU_MODEL
logger.info(f"Environment loaded. Using model: {OPENAI_NLU_MODEL}")



def create_app() -> FastAPI:
    load_dotenv()  # read .env into environment (OPENAI_API_KEY, etc.)
    logger.info("starting creation for FastAPI...")
    app = FastAPI(title="SymptSpher API", version="1.0")

    try:
        # CORS: allow Streamlit origin to call our API
        allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        logger.info(f"CORS configured with allowed origins: {allowed}")

        # Singletons: load once, reuse across requests
        app.state.codebook = Codebook("data/release_evidences.json", "data/release_conditions.json")
        logger.info("Codebook loaded successfully.")
        app.state.nlu = OpenAINLU()
        logger.info("OpenAINLU initialized successfully.")
        app.state.store = SessionStore("cases/sample_cases.jsonl")  # swap to DB later
        logger.info("SessionStore initialized successfully.")

        # Routers
        app.include_router(patient.router, prefix="/v1/patient", tags=["patient"])
        app.include_router(professor.router, prefix="/v1/professor", tags=["professor"])
        logger.info("Routers 'patient' and 'professor' registered successfully.")

    except Exception as e:
        logger.error(f"Error during app creation: {e}")
    
    return app

    # Uvicorn will import this symbol
app = create_app()

@app.get("/health")

def health():
    import os
    key = os.getenv("OPENAI_API_KEY") or ""
    logger.info("Health check accessed.")
    return {"status":"ok","has_key":bool(key), "prefix": key[:3]}