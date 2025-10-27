# store.py — minimal in-memory store. Replace with Postgres later (sessions/turns/grades).

import json, random
from pathlib import Path
from typing import Dict, Set

class SessionStore:
    def __init__(self, cases_path: str):
        # Load JSONL demo cases (one JSON object per line)
        txt = Path(cases_path).read_text(encoding="utf-8").strip()
        self.cases = [json.loads(l) for l in txt.splitlines() if l.strip()]

        # session_id -> { "case": {...}, "revealed": set([...]), "turns": int }
        self.state: Dict[str, dict] = {}

    def start(self, sid: str) -> dict:
        """Pick a random case; seed 'revealed' with INITIAL_EVIDENCE."""
        case = random.choice(self.cases)
        self.state[sid] = {
            "case": case,
            "revealed": set([case["initial_evidence"]]),
            "turns": 0
        }
        return case

    def get(self, sid: str) -> dict | None:
        return self.state.get(sid)

    def reveal(self, sid: str, token: str) -> None:
        self.state[sid]["revealed"].add(token)

    def inc_turn(self, sid: str) -> None:
        self.state[sid]["turns"] += 1

    def revealed(self, sid: str) -> Set[str]:
        return set(self.state[sid]["revealed"])

    def turns(self, sid: str) -> int:
        return int(self.state[sid]["turns"])
