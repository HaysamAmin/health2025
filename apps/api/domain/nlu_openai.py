# nlu_openai.py — OpenAI-backed NLU that maps a student's question
# to a DDXPlus evidence head (E_**) and optional value (V_** or integer).
# ------------------------------------------------------------------------------
# Uses Chat Completions with function/tool calling to force structured JSON:
#   { "feature": "E_53", "value": null }                 # binary
#   { "feature": "E_55", "value": "V_167" }              # categorical
#   { "feature": "E_56", "value": 6 }                    # numeric/ordinal
#
# Robustness:
# - Forces function call (tool_choice) so we always get structured output.
# - Parses JSON from call.function.arguments (SDK v1.x).
# - Keyword fallback for core intents (cough/fever/sore throat/runny nose/pain)
#   so the system never returns UNKNOWN for standard questions.
# ------------------------------------------------------------------------------

from __future__ import annotations
import json
from typing import Optional, Tuple, Union
from openai import OpenAI
from conf.openapi_config import OPENAI_API_KEY, OPENAI_NLU_MODEL
from openai import OpenAI

_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = OPENAI_NLU_MODEL

# Function schema the model must call
NLU_TOOL = [{
    "type": "function",
    "function": {
        "name": "map_to_evidence",
        "description": "Map clinical question to DDXPlus evidence head (E_**) and optional value (V_** or integer).",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "E_* code like E_53 (cough), E_55 (pain location), E_56 (pain intensity)."
                },
                "value": {
                    "anyOf": [
                        {"type": "string",  "description": "V_* code for categorical values"},
                        {"type": "integer", "description": "Integer for ordinal features"},
                        {"type": "null"}
                    ]
                }
            },
            "required": ["feature"]
        }
    }
}]

SYSTEM_PROMPT = (
    "You are a precise mapper from English clinical questions to DDXPlus evidence codes.\n"
    "Return only:\n"
    "  - feature: E_** head that best matches the user's question\n"
    "  - value:   V_** code (if categorical), integer (if ordinal), or null\n"
    "Avoid free text. If the question is about generic pain presence, set feature='PAIN_ANY' and value=null."
)

# A couple of few-shots to anchor the mapping behavior
FEW_SHOTS = [
    {"role":"user","content":"Do you have a cough?"},
    {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"map_to_evidence","arguments": json.dumps({"feature":"E_53","value":None})}}]},
    {"role":"tool","tool_call_id":"c1","name":"map_to_evidence","content": json.dumps({"feature":"E_53","value":None})},

    {"role":"user","content":"Any fever recently?"},
    {"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"map_to_evidence","arguments": json.dumps({"feature":"E_91","value":None})}}]},
    {"role":"tool","tool_call_id":"c2","name":"map_to_evidence","content": json.dumps({"feature":"E_91","value":None})},

    {"role":"user","content":"Do you have pain anywhere?"},
    {"role":"assistant","tool_calls":[{"id":"c3","type":"function","function":{"name":"map_to_evidence","arguments": json.dumps({"feature":"PAIN_ANY","value":None})}}]},
    {"role":"tool","tool_call_id":"c3","name":"map_to_evidence","content": json.dumps({"feature":"PAIN_ANY","value":None})},
]

# Minimal keyword fallback so common intents NEVER fail (tune codes to your codebook)
KW_FALLBACK = {
    "cough": "E_53",
    "fever": "E_91",
    "sore throat": "E_97",
    "throat pain": "E_97",
    # TODO: set to the correct E_* for rhinorrhea/runny nose in YOUR codebook
    "runny nose": "E_201",          # ← replace if your codebook uses a different E_*
    "nasal discharge": "E_201",
    # Generic pain
    "pain": "PAIN_ANY",
}

class OpenAINLU:
    """Return (feature_head, value_or_None) for a student's natural-language question."""

    def parse(self, text: str) -> Tuple[str, Optional[Union[str, int]]]:
        # 1) Primary path: tool calling with structured JSON
        try:
            messages = [{"role":"system","content":SYSTEM_PROMPT}] + FEW_SHOTS + [
                {"role":"user","content": text}
            ]

            resp = _client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=NLU_TOOL,
                # Force the model to call our function and NOT answer in plain text
                tool_choice={"type":"function","function":{"name":"map_to_evidence"}},
                temperature=0
            )

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            if tool_calls:
                call = tool_calls[0]
                if call.function.name == "map_to_evidence":
                    # Parse JSON string from arguments (SDK v1.x)
                    args = json.loads(call.function.arguments or "{}")
                    feature = args.get("feature") or "UNKNOWN"
                    value = args.get("value", None)
                    if feature != "UNKNOWN":
                        return feature, value
            # If no tool call (unexpected), drop to fallback
        except Exception as e:
            # Log but do not crash the API; we'll try keyword fallback next
            print("NLU error:", repr(e))

        # 2) Fallback path: simple keyword map to guarantee usability
        t = text.lower()
        for k, feat in KW_FALLBACK.items():
            if k in t:
                return feat, None

        # 3) Last resort
        return "UNKNOWN", None
