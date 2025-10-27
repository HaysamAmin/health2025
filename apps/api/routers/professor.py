# professor.py — inside grade()

from ..domain.codebook import Codebook
from fastapi import APIRouter, Request, HTTPException
from ..models.schema import GradeReq, GradeResp
from ..domain.scoring import diagnosis_credit, per_score, il_score

router = APIRouter()

@router.post("/grade", response_model=GradeResp)
async def grade(req: Request, body: GradeReq):
    app = req.app
    store = app.state.store
    cb: Codebook = app.state.codebook          # <-- get codebook

    sess = store.get(body.session_id)
    if not sess:
        raise HTTPException(404, "Session not found.")

    case = sess["case"]
    normalized = body.diagnosis_text.strip()

    credit = diagnosis_credit(case["differential"], normalized)
    per = per_score(case["evidences"], store.revealed(body.session_id))
    il = il_score(store.turns(body.session_id))
    score = int(round(0.6 * credit + 0.3 * per + 0.1 * max(0, 100 - il)))

    # Build feedback on *heads*, then DECODE them to human text
    heads_in_case = {ev.split("_@_")[0] for ev in case["evidences"]}
    revealed_heads = {ev.split("_@_")[0] for ev in store.revealed(body.session_id)}
    missed_heads = list(heads_in_case - revealed_heads)[:3]

    # 🔧 Decode E_* heads into readable questions
    decoded_missed = [cb.decode_token(h) for h in missed_heads]
    feedback = [f"Consider asking about: {dm}" for dm in decoded_missed] if decoded_missed else ["Good coverage."]

    return GradeResp(
        normalized_dx=normalized,
        credit=credit, per=per, il=il, score=score,
        feedback=feedback
    )
