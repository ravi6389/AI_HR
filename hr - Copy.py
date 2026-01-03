import streamlit as st
import pdfplumber
import docx
import os
import json
import pandas as pd
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="AI Resume Ranking + Chatbot", layout="wide")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# st.write('GROQ API Key is..', GROQ_API_KEY)

llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")


# ----------------------------
# UTILS
# ----------------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def groq_llm(prompt: str) -> str:
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    return response.content

# ----------------------------
# JD PARSER
# ----------------------------
def parse_jd(jd_text):
    prompt = f"""
Extract structured hiring requirements from this JD.

Return ONLY valid JSON.
Do not add explanation or markdown.

JSON schema:
{{
  "core_skills": [],
  "nice_to_have": [],
  "minimum_experience_years": 0,
  "role_level": ""
}}

JD:
{jd_text}
"""
    raw = groq_llm(prompt)
    # st.write("🔍 RAW JD LLM OUTPUT:", raw)   # TEMP DEBUG
    return json.loads(raw)

import re

def safe_json_loads(text: str):
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")

    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = text.replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in:\n{text}")

    return json.loads(match.group())

# ----------------------------
# RESUME PARSER
# ----------------------------
def parse_resume(resume_text):
    prompt = f"""
You are an HR parsing system.

Return ONLY valid JSON.
No explanation. No markdown.

JSON schema:
{{
  "summary": "",
  "skills": {{}},
  "recent_tools": [],
  "experience_years": 0,
  "notable_projects": []
}}

Resume:
{resume_text}
"""
    raw = groq_llm(prompt)
    return safe_json_loads(raw)



# ----------------------------
# SCORING
# ----------------------------
def score_candidate(candidate, jd):
    skill_match = len(
        set(candidate["skills"].keys()) & set(jd["core_skills"])
    ) / max(len(jd["core_skills"]), 1)

    experience_score = min(
        candidate.get("experience_years", 0) / max(jd["minimum_experience_years"], 1), 1
    )

    final_score = round(
        (0.6 * skill_match + 0.4 * experience_score) * 100, 2
    )

    return final_score

# ----------------------------
# CHAT ENGINE
# ----------------------------
def chat_answer(question, candidates, ranking_df, jd):
    context = f"""
You are a hiring analyst AI.

Job Requirements:
{json.dumps(jd, indent=2)}

Candidate Profiles:
{json.dumps(candidates, indent=2)}

Ranking Table:
{ranking_df.to_string(index=False)}

Rules:
- Use ONLY the provided data
- Do NOT hallucinate
- Explain reasoning clearly
"""

    prompt = f"{context}\n\nQuestion:\n{question}"
    return groq_llm(prompt)

# ============================
# STREAMLIT UI
# ============================

st.title("🧠 AI Resume Ranking + Recruiter Chatbot")

with st.sidebar:
    st.header("📂 Upload Files")
    resumes = st.file_uploader(
        "Upload Resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx"]
    )

if resumes and jd_file:
    with st.spinner("Parsing Job Description..."):
        jd_text = extract_text(jd_file)
        jd = parse_jd(jd_text)

    st.subheader("📄 Job Requirements")
    st.json(jd)

    candidates = {}
    rows = []

    with st.spinner("Parsing Resumes & Ranking..."):
        for idx, resume in enumerate(resumes):
            text = extract_text(resume)
            parsed = parse_resume(text)
            score = score_candidate(parsed, jd)

            cid = f"C{idx+1}"
            candidates[cid] = parsed

            rows.append({
                "Candidate": cid,
                "Score": score,
                "Skills": ", ".join(parsed["skills"].keys())
            })

    ranking_df = pd.DataFrame(rows).sort_values("Score", ascending=False)

    # st.subheader("📊 Ranked Candidates")
    # st.dataframe(ranking_df, use_container_width=True)

    # ----------------------------
    # CHAT
    # ----------------------------
    st.subheader("💬 Recruiter Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_q = st.chat_input("Ask about candidates, comparisons, or scores...")

    if user_q:
        st.chat_message("user").write(user_q)
        answer = chat_answer(user_q, candidates, ranking_df, jd)
        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_q}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👈 Upload resumes and a job description to begin.")




