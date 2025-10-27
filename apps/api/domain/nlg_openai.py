# nlg_openai.py — Humanize patient answers using OpenAI, grounded in case facts.
# --------------------------------------------------------------------------------
# Purpose:
#   Given a symptom head (E_**), the student's question text, and the case’s
#   evidence tokens, produce a short, first-person patient response (yes/no with
#   relevant details). We pass ONLY the relevant decoded evidence as context.
#
# Safety:
#   - Grounded: the model only sees decoded facts for the asked head (and related heads).
#   - Deterministic: temperature=0.
#   - Bounded: short answers (<= 25 words), polite, no extra info.
#
# Fallback:
#   If the OpenAI call fails, we return a rule-based answer.
# --------------------------------------------------------------------------------

from __future__ import annotations
import os, json
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_NLU_MODEL", "gpt-4o-mini")

# Pain-related heads to stitch richer answers
PAIN_HEADS = {"E_54","E_55","E_56","E_57","E_58","E_59"}

SYSTEM_PROMPT = (
    "You are a simulated patient. Answer in first person, concise (<= 25 words), "
    "polite, and consistent ONLY with the facts provided. If the symptom is absent, say 'No.' "
    "If present, say 'Yes' and include the most relevant details. Do not invent data."
)

FEW_SHOTS = [
    # Cough present
    {"role":"user","content":json.dumps({
        "question":"Do you have a cough?",
        "symptom_head":"E_53",
        "present":True,
        "facts":["Cough"],
        "detail_facts":[]
    })},
    {"role":"assistant","content":"Yes."},

    # Fever absent
    {"role":"user","content":json.dumps({
        "question":"Any fever recently?",
        "symptom_head":"E_91",
        "present":False,
        "facts":[],
        "detail_facts":[]
    })},
    {"role":"assistant","content":"No."},

    # Pain with details
    {"role":"user","content":json.dumps({
        "question":"Do you have pain?",
        "symptom_head":"E_55",
        "present":True,
        "facts":["Pain present"],
        "detail_facts":["Location → left arm","Character → sharp","Intensity → 6"]
    })},
    {"role":"assistant","content":"Yes, sharp pain in my left arm, about 6 out of 10."},
]

def _collect_related_facts(symptom_head: str, case_evidences: List[str], decode) -> Dict[str, List[str]]:
    """
    Select only the tokens relevant to this head, decode them to English.
    For pain, include related heads for richer answers.
    """
    heads_needed = {symptom_head}
    if symptom_head in PAIN_HEADS or symptom_head == "PAIN_ANY":
        heads_needed |= PAIN_HEADS  # add all pain facets

    present_tokens = [ev for ev in case_evidences if ev.split("_@_")[0] in heads_needed]
    decoded = [decode(t) for t in present_tokens]

    # Split into minimal 'facts' and more specific 'detail_facts'
    detail_facts = []
    facts = []
    for s in decoded:
        if " → " in s:
            detail_facts.append(s.replace(" -> ", " → "))  # normalize arrow
        else:
            facts.append(s)
    return {"facts": facts, "detail_facts": detail_facts}

def _rule_fallback(present: bool, detail_facts: List[str]) -> str:
    """Simple, safe fallback sentence if the API call fails."""
    if not present:
        return "No."
    # Try to build a short detail string from decoded facts
    # e.g., ["Where is the pain? → left arm","How intense → 6"]
    bits = []
    for d in detail_facts:
        try:
            _, right = d.split("→", 1)
            bits.append(right.strip())
        except ValueError:
            bits.append(d)
    if not bits:
        return "Yes."
    # keep it tight
    return "Yes, " + ", ".join(bits)[:80].rstrip(", ") + "."

def human_answer(question_text: str,
                 symptom_head: str,
                 present: bool,
                 case_evidences: List[str],
                 decode) -> str:
    """
    Compose a human-style answer using OpenAI with strict grounding.
    - question_text: student's question
    - symptom_head: E_** (or "PAIN_ANY")
    - present: whether the symptom head is present in case
    - case_evidences: all tokens for the case
    - decode: function to decode tokens
    """
    rel = _collect_related_facts(symptom_head, case_evidences, decode)
    payload = {
        "question": question_text,
        "symptom_head": symptom_head,
        "present": present,
        "facts": rel["facts"],            # minimal claims (strings)
        "detail_facts": rel["detail_facts"]  # specific facets (strings)
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,          # deterministic for assessment
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                *FEW_SHOTS,
                {"role":"user","content":json.dumps(payload)}
            ]
        )
        ans = (resp.choices[0].message.content or "").strip()
        # Guardrails: very short, no invented content
        if not ans:
            return _rule_fallback(present, rel["detail_facts"])
        if len(ans.split()) > 25:            # enforce brevity
            ans = " ".join(ans.split()[:25])
        return ans
    except Exception as e:
        print("NLG error:", repr(e))
        return _rule_fallback(present, rel["detail_facts"])
