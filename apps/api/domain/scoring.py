# scoring.py — scoring primitives: Diagnosis Credit, PER, IL

from typing import List, Dict, Set

def diagnosis_credit(differential: List[Dict], dx: str) -> int:
    """
    differential example: [{"disease":"Influenza (flu)","prob":0.55}, ...]
    dx: student's final diagnosis (already normalized or trusted as-is)
    """
    p = 0.0
    for d in differential:
        if d["disease"].lower() == dx.lower():
            p = d["prob"]
            break
    return int(round(100 * p))

def per_score(case_evidences: List[str], revealed: Set[str]) -> int:
    """
    PER = positive evidence recall
    Compare feature heads present in the case vs. heads the student revealed.
    """
    case_heads = {ev.split("_@_")[0] for ev in case_evidences}
    rev_heads  = {rv.split("_@_")[0] for rv in revealed}
    if not case_heads:
        return 0
    return int(round(100 * len(case_heads & rev_heads) / len(case_heads)))

def il_score(turns_used: int) -> int:
    """Return interaction length as a raw count (you can map this to a score later)."""
    return max(0, turns_used)
