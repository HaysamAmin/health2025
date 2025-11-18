# patient.py — Patient-facing endpoints using OpenAI NLU (nlu_openai) for intent,
# and OpenAI NLG (nlg_openai) for HUMAN answers grounded in the current case.
# ---------------------------------------------------------------------------------
# Endpoints
#   POST /v1/patient/start : start a new session (pick a random case)
#   POST /v1/patient/ask   : map student's question → E_* head [+ value], then
#                            answer with a short, first-person sentence consistent
#                            ONLY with the facts present in the current case.
#
# Notes
# - We reveal tokens to the session store so grading (PER/IL) works.
# - "PAIN_ANY" is a special pseudo-feature used to answer generic pain questions
#   by aggregating pain facets (E_54..E_59) if present.
# ---------------------------------------------------------------------------------

from fastapi import APIRouter, Request, HTTPException
from ..models.schema import StartReq, StartResp, AskReq, AskResp
from ..domain.codebook import Codebook
from ..domain.nlu_openai import OpenAINLU
from ..domain.nlg_openai import human_answer, PAIN_HEADS  # NEW: humanized replies
from conf.log_config import logger

router = APIRouter()

def _present_for_head(head: str, case_evidences: list[str]) -> bool:
    """Return True if any token with this E_* head exists in the case."""
    return any(head == ev.split("_@_")[0] for ev in case_evidences)


@router.post("/start", response_model=StartResp)
async def start(req: Request, body: StartReq):
    """
    Start/reset a session:
      - Pick a case
      - Seed 'revealed' with the case's INITIAL_EVIDENCE
      - Return demographics + decoded initial clue
    """
    store = req.app.state.store
    cb: Codebook = req.app.state.codebook
    case = store.start(body.session_id)
    return {
        "age": case["age"],
        "sex": case["sex"],
        "initial_evidence": cb.decode_token(case["initial_evidence"]),
        "case_id": case.get("id")
    }


@router.post("/ask", response_model=AskResp)
async def ask(req: Request, body: AskReq):
    """
    Handle a student's question using:
      1) OpenAI NLU (map text → feature head E_* [+ value])
      2) Evidence check versus the case
      3) OpenAI NLG for a short, first-person patient answer grounded in facts
    """
    logger.info(f"Received /ask call: session_id={body.session_id}, text={body.text}")
    try:
        app = req.app
        store = app.state.store
        cb: Codebook = app.state.codebook
        nlu: OpenAINLU = app.state.nlu
        sess = store.get(body.session_id)

        if not sess:
            raise HTTPException(404, "Session not found. Call /v1/patient/start first.")
        case = sess["case"]

        # Debug: log before parsing
        logger.info("Calling NLU.parse() ...")
        feature, value = nlu.parse(body.text)
        logger.info(f"NLU result: feature={feature}, value={value}")

        # ---- Step 1: NLU → (feature head, optional value) ------------------------
        f_head = feature

        # ---- Step 1a: Special handling for generic pain questions ----------------
        if f_head == "PAIN_ANY":
            # Determine if any pain facets are present in this case.
            present = any(ev.split("_@_")[0] in PAIN_HEADS for ev in case["evidences"])
            # Optionally reveal all present pain tokens (so PER captures them).
            if present:
                for ev in case["evidences"]:
                    if ev.split("_@_")[0] in PAIN_HEADS:
                        store.reveal(body.session_id, ev)

            store.inc_turn(body.session_id)
            revealed = sorted(store.revealed(body.session_id))
            decoded = [cb.decode_token(t) for t in revealed]

            # Human, first-person answer grounded in the available pain facts
            answer = human_answer(body.text, "PAIN_ANY", present, case["evidences"], cb.decode_token)
            return AskResp(answer=answer, revealed=revealed, decoded=decoded)

        # ---- Step 2: Compose token for value-carrying features -------------------
        token = f_head
        if value is not None:
            if isinstance(value, int):
                token = f"{f_head}_@_{value}"
            elif isinstance(value, str):
                token = f"{f_head}_@_{value if value.startswith('V_') else 'V_'+value}"

        # Is this feature present in the case? (exact token OR by head)
        has_feature = any(token == ev or f_head == ev.split("_@_")[0] for ev in case["evidences"])

        # ---- Step 3: Update session, reveal, and produce a HUMAN answer ----------
        store.inc_turn(body.session_id)

        if has_feature:
            # Reveal at least the head the student asked (we reveal composed token)
            store.reveal(body.session_id, token)

            revealed = sorted(store.revealed(body.session_id))
            decoded  = [cb.decode_token(t) for t in revealed]

            present = _present_for_head(f_head, case["evidences"])  # True here, but keep consistent API
            # Human, short, first-person answer grounded in case facts for this head
            answer = human_answer(body.text, f_head, present, case["evidences"], cb.decode_token)
            return AskResp(answer=answer, revealed=revealed, decoded=decoded)

        # Not present → concise, human "No." (still grounded; NLG enforces brevity)
        revealed = sorted(store.revealed(body.session_id))
        decoded  = [cb.decode_token(t) for t in revealed]
        answer   = human_answer(body.text, f_head, False, case["evidences"], cb.decode_token)
        return AskResp(answer=answer, revealed=revealed, decoded=decoded)

    except Exception as e:
        logger.exception(f"Error inside /ask: {e}")
        # Always return a valid AskResp-like structure even when failing
        return {
            "answer": None,
            "decoded": [],
            "revealed": [],
            "error": str(e)
        }
