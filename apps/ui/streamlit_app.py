# Streamlit UI — student-facing app that calls FastAPI.
# Shows demographics + initial clue, lets the student ask questions,
# and submits a final diagnosis to be graded.

import streamlit as st, requests, uuid

API = "http://127.0.0.1:8000"  # FastAPI base URL

st.set_page_config(page_title="SymptSpher (OpenAI NLU)", layout="centered")
st.title("SymptSpher — OpenAI NLU")

# Start new session on first load
if "sid" not in st.session_state:
    st.session_state.sid = f"sess-{uuid.uuid4()}"
    r = requests.post(f"{API}/v1/patient/start", json={"session_id": st.session_state.sid}, timeout=15)
    r.raise_for_status()
    st.session_state.meta = r.json()

# Display demographics + initial evidence
st.write(f"**Patient:** {st.session_state.meta['age']} y/o, {st.session_state.meta['sex']}")
st.write(f"**Initial clue:** {st.session_state.meta['initial_evidence']}")

# Ask a question
q = st.text_input("Ask the patient:")
if st.button("Send") and q.strip():
    r = requests.post(f"{API}/v1/patient/ask",
                      json={"session_id": st.session_state.sid, "text": q},
                      timeout=20)
    if r.ok:
        data = r.json()
        st.write("**Patient:**", data["answer"])
        if data.get("decoded"):
            with st.expander("Revealed findings (decoded)"):
                st.write("\n".join(f"- {d}" for d in data["decoded"]))
    else:
        st.error(f"Error: {r.status_code} {r.text}")

# Submit final diagnosis
st.subheader("Submit your diagnosis")
dx = st.text_input("Diagnosis (free text):")
if st.button("Grade me") and dx.strip():
    r = requests.post(f"{API}/v1/professor/grade",
                      json={"session_id": st.session_state.sid, "diagnosis_text": dx},
                      timeout=20)
    if r.ok:
        g = r.json()
        st.success(f"Normalized: {g['normalized_dx']}")
        st.write(f"**Diagnosis Credit:** {g['credit']}")
        st.write(f"**PER:** {g['per']}%")
        st.write(f"**IL (turns):** {g['il']}")
        st.write(f"**Final Score:** {g['score']}")
        st.write("**Feedback:**")
        for tip in g["feedback"]:
            st.write(f"- {tip}")
    else:
        st.error(f"Error: {r.status_code} {r.text}")

st.caption("Training tool — not medical advice.")
