# SymptomSphere – MLOps Service Contracts
(18 Total – 6 per Use Case)

---

# USE CASE 1 — Virtual Patient Chatbot (6 Service Contracts)

## 1. Service Contract: StartSession
**Description:** Initializes a new patient simulation session and returns demographics + initial evidence.  
**Endpoint:** POST /v1/patient/start  
**Inputs:**  
- student_id (string)  
- case_id (int)  

**Outputs:**  
- session_id (uuid)  
- demographics (object)  
- initial_clue (string)

**Constraints:**  
- Must load case data from DDXPlus  
- Must log start/end  

---

## 2. Service Contract: MapQuestionToEvidence
**Description:** Converts student question → evidence token (E_*).  
**Endpoint:** POST /v1/patient/ask  
**Inputs:**  
- session_id (uuid)  
- message (string)

**Outputs:**  
- evidence_code (E_xxx)  
- value (string/null)  
- response_text (string)

**Constraints:**  
- temperature=0  
- Must use OpenAI NLU tool-calling  

---


# USE CASE 2 — Diagnostic Skill Development (6 Service Contracts)

## 3. Service Contract: CalculateCompositeScore
**Description:** Produces composite CPS score from DC+PER+IL weighted sum.  
**Endpoint:** POST /v1/professor/grade  
**Inputs:**  
- dc  
- per  
- il  

**Outputs:**  
- composite_score  
- feedback_text

---


# ✔ All 3 service contracts completed.
# ✔ Ready for submission.
# ✔ Place this file in: /docs/service_contracts.md
