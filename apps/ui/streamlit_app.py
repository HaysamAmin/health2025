# Streamlit UI — student-facing app that calls FastAPI.
# Shows demographics + initial clue, lets the student ask questions,
# and submits a final diagnosis to be graded.

import streamlit as st, requests, uuid
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from conf.log_config import logger


# API = "http://127.0.0.1:8000"  # FastAPI base URL. Use this for local testing.
API = "http://fastapi:8000"

st.set_page_config(page_title="SymptSpher (OpenAI NLU)", layout="centered")
st.title("SymptSpher — OpenAI NLU")

# === LOG: firt load ===
logger.info("Streamlit Application loaded successfully.")

try:
    # Start new session on first load
    if "sid" not in st.session_state:
        st.session_state.sid = f"sess-{uuid.uuid4()}"
        logger.info(f"Creating a new session: {st.session_state.sid}")
        r = requests.post(
            f"{API}/v1/patient/start",
            json={"session_id": st.session_state.sid},
            timeout=15
        )
        r.raise_for_status()
        st.session_state.meta = r.json()
        logger.info(f"Session {st.session_state.sid} started with metadata: {st.session_state.meta}")
    else:
        logger.info(f"Resuming existing session: {st.session_state.sid}")

    # Display demographics + initial evidence
    st.write(f"**Patient:** {st.session_state.meta['age']} y/o, {st.session_state.meta['sex']}")
    st.write(f"**Initial clue:** {st.session_state.meta['initial_evidence']}")

    # Ask a question
    q = st.text_input("Ask the patient:")
    if st.button("Send") and q.strip():
        logger.info(f"Session {st.session_state.sid} asking question: {q}")
        r = requests.post(
            f"{API}/v1/patient/ask",
            json={"session_id": st.session_state.sid, "text": q},
            timeout=20
        )

        if r.ok:
            data = r.json()
            logger.info(f"Session {st.session_state.sid} received response: {data}")
            answer = data.get("answer")
            if answer:
                st.write("**Patient:**", answer)
            if data.get("decoded"):
                with st.expander("Revealed findings (decoded)"):
                    st.write("\n".join(f"- {d}" for d in data["decoded"]))
        else:
            logger.warning(f"Error in /v1/patient/ask: {r.status_code} {r.text}")
            st.error(f"Error: {r.status_code} {r.text}")

    # Submit final diagnosis
    st.subheader("Submit your diagnosis")
    dx = st.text_input("Diagnosis (free text):")
    if st.button("Grade me") and dx.strip():
        logger.info(f"submitting diagnosis for grading: {dx}")
        r = requests.post(f"{API}/v1/professor/grade",
                        json={"session_id": st.session_state.sid, "diagnosis_text": dx},
                        timeout=20)
        if r.ok:
            g = r.json()
            logger.info(f"Received grading results for session {st.session_state.sid}")
            st.success(f"Normalized: {g['normalized_dx']}")
            st.write(f"**Diagnosis Credit:** {g['credit']}")
            st.write(f"**PER:** {g['per']}%")
            st.write(f"**IL (turns):** {g['il']}")
            st.write(f"**Final Score:** {g['score']}")
            st.write("**Feedback:**")
            for tip in g["feedback"]:
                st.write(f"- {tip}")
        else:
            logger.warning(f"Error in /v1/professor/grade: {r.status_code} {r.text}")
            st.error(f"Error: {r.status_code} {r.text}")
        
except requests.exceptions.RequestException as e:
    logger.error(f"RequestException: {e}")
    st.error(f"An error occurred while communicating with the server: {e}")

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    st.error(f"An unexpected error occurred: {e}")

st.caption("Training tool — not medical advice.")
