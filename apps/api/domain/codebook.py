# codebook.py — DDXPlus codebook loader & decoder (robust format support)
# -----------------------------------------------------------------------------
# Purpose
#   Load the DDXPlus codebooks and provide helpers to decode evidence tokens
#   (e.g., "E_91", "E_55_@_V_167", "E_56_@_5") into human-readable text.
#
# Inputs (paths you pass to Codebook(...)):
#   - release_evidences.json   → features (E_*), data_type (B/C/M), question_en, possible-values (V_*)
#   - release_conditions.json  → diseases catalog (ids, names, ICD-10, etc.)  # not required for decoding
#
# Robustness
#   - Supports multiple JSON shapes: list, dict with "evidences"/"data" key, dict-of-dicts ({"E_91": {...}}),
#     or JSON Lines (JSONL; one JSON per line).
#
# Exposes
#   - decode_token(token: str) -> str
#       "E_91"              → "Do you have a fever?"              (example)
#       "E_55_@_V_167"      → "Where is the pain? → temple (L)"
#       "E_56_@_5"          → "How intense is the pain? → 5"
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class Codebook:
    """Load evidence/condition codebooks and build fast lookup maps."""

    # Public attributes you might find useful elsewhere
    e: List[Dict[str, Any]]                 # normalized evidences list
    c: Any                                  # raw conditions payload (shape not enforced)
    E_MAP: Dict[str, Dict[str, Any]]        # "E_XX" → evidence row
    V_MAP: Dict[Tuple[str, str], Dict[str, Any]]  # ("E_XX","V_YYY") → value row

    def __init__(self, evidences_path: str, conditions_path: str):
        # --- Load evidences (robust to list / dict / dict-of-dicts / JSONL) ---
        txt_e = Path(evidences_path).read_text(encoding="utf-8").strip()
        try:
            e_obj = json.loads(txt_e)  # Try plain JSON first
        except json.JSONDecodeError:
            # Fallback: JSON Lines (one JSON object per line)
            e_obj = [json.loads(line) for line in txt_e.splitlines() if line.strip()]

        # Normalize to a list[dict]
        if isinstance(e_obj, dict):
            # Case A: dict with list under a common key
            for k in ("evidences", "evidence", "data", "items"):
                if isinstance(e_obj.get(k), list):
                    self.e = e_obj[k]  # type: ignore[index]
                    break
            else:
                # Case B: dict-of-dicts, e.g. {"E_91": {...}, "E_53": {...}}
                self.e = []
                for code, row in e_obj.items():
                    if isinstance(row, dict):
                        row = {"code_evidence": code, **row}  # backfill code if missing
                        self.e.append(row)
        elif isinstance(e_obj, list):
            self.e = e_obj
        else:
            raise ValueError("Unexpected evidences JSON format (not list/dict/JSONL).")

        # --- Load conditions (kept for reference; not required for decode) ---
        txt_c = Path(conditions_path).read_text(encoding="utf-8").strip()
        try:
            self.c = json.loads(txt_c)
        except json.JSONDecodeError:
            self.c = [json.loads(line) for line in txt_c.splitlines() if line.strip()]

        # --- Build lookups ---
        self._build_maps()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _build_maps(self) -> None:
        """Build evidence and value maps with normalized field names."""
        # Map "E_XX" → evidence row
        self.E_MAP = {}
        for ev in self.e:
            if not isinstance(ev, dict):
                continue
            code = ev.get("code_evidence") or ev.get("code") or ev.get("id")
            if code:
                ev["code_evidence"] = code  # normalize
                self.E_MAP[code] = ev

        # Map ("E_XX","V_YYY") → value row
        self.V_MAP = {}
        for ev in self.E_MAP.values():
            vals = (
                ev.get("possible-values")
                or ev.get("possible_values")
                or ev.get("values")
                or []
            )
            if not isinstance(vals, list):
                continue
            for val in vals:
                if not isinstance(val, dict):
                    continue
                vcode = val.get("code_value") or val.get("code") or val.get("id")
                if vcode:
                    self.V_MAP[(ev["code_evidence"], vcode)] = val

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def decode_token(self, token: str) -> str:
        """
        Convert evidence tokens into human-readable text using the codebook.

        Supported token shapes:
          - 'E_XX'                  (binary present)
          - 'E_XX_@_V_YYY'          (categorical / multi-choice value)
          - 'E_XX_@_<number>'       (numeric / ordinal value)

        Returns a best-effort readable string; falls back to the original token
        if a code cannot be found.
        """
        # Case 1: simple binary token 'E_XX'
        if "_@_" not in token:
            ev = self.E_MAP.get(token)
            # Prefer English question/label if present
            for k in ("question_en", "name_en", "label_en", "question", "name", "label"):
                if ev and k in ev:
                    return str(ev[k])
            return token  # unknown code → leave as-is

        # Case 2: token has feature head and a value tail: 'E_XX_@_<tail>'
        head, tail = token.split("_@_", 1)
        ev = self.E_MAP.get(head)

        # 2a) categorical/multi-choice value 'V_YYY'
        if tail.startswith("V_"):
            val = self.V_MAP.get((head, tail))
            if ev and val:
                q = ev.get("question_en") or ev.get("name_en") or ev.get("question") or head
                label = (
                    val.get("value_en")
                    or val.get("label_en")
                    or val.get("value")
                    or val.get("label")
                    or tail
                )
                return f"{q} → {label}"

        # 2b) numeric/ordinal value: not starting with 'V_'
        if ev:
            q = ev.get("question_en") or ev.get("name_en") or ev.get("question") or head
            return f"{q} → {tail}"

        # Fallback if code wasn’t found
        return token


# -----------------------------------------------------------------------------
# Minimal usage example (remove in production if you like)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust paths to your repo layout
    cb = Codebook(
        evidences_path="data/release_evidences.json",
        conditions_path="data/release_conditions.json",
    )
    print(cb.decode_token("E_91"))           # e.g., "Do you have a fever?"
    print(cb.decode_token("E_55_@_V_167"))   # e.g., "Where is the pain? → temple (L)"
    print(cb.decode_token("E_56_@_5"))       # e.g., "How intense is the pain? → 5"
