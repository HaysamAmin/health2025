# schema.py — Pydantic request/response models for FastAPI

from pydantic import BaseModel
from typing import List, Optional

class StartReq(BaseModel):
    session_id: str

class StartResp(BaseModel):
    age: int
    sex: str
    initial_evidence: str
    case_id: Optional[str] = None

class AskReq(BaseModel):
    session_id: str
    text: str

class AskResp(BaseModel):
    answer: str
    revealed: List[str]
    decoded: List[str]

class GradeReq(BaseModel):
    session_id: str
    diagnosis_text: str

class GradeResp(BaseModel):
    normalized_dx: str
    credit: int
    per: int
    il: int
    score: int
    feedback: List[str]
