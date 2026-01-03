import streamlit as st
import fitz  # PyMuPDF
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI-Powered Resume Analyzer",
    layout="centered"
)

st.title("📄 AI-Powered Resume Analyzer")
st.caption(
    "ATS-style Resume Intelligence System using NLP, TF-IDF, and Explainable Scoring"
)

# ===================== NLP SETUP =====================
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()

# ===================== TEXT PROCESSING =====================
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ===================== CORE ML LOGIC =====================
def semantic_similarity(resume: str, jd: str):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000
    )
    tfidf_matrix = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100, 2), vectorizer.get_feature_names_out(), tfidf_matrix.toarray()

def keyword_gap_analysis(features, tfidf_matrix):
    resume_vec, jd_vec = tfidf_matrix

    df = pd.DataFrame({
        "Keyword": features,
        "Resume_Weight": resume_vec,
        "JD_Weight": jd_vec
    })

    matched = df[(df.Resume_Weight > 0) & (df.JD_Weight > 0)]
    missing = df[(df.JD_Weight > 0) & (df.Resume_Weight == 0)]

    return (
        matched.sort_values("JD_Weight", ascending=False),
        missing.sort_values("JD_Weight", ascending=False)
    )

# ===================== ATS SIGNALS =====================
def experience_signal_score(text: str) -> float:
    experience_terms = [
        "experience", "project", "developed", "implemented",
        "designed", "led", "built", "optimized", "deployed",
        "analyzed", "engineered"
    ]
    hits = sum(1 for term in experience_terms if term in text)
    return round(min(hits / len(experience_terms), 1.0) * 100, 2)

def skill_coverage_score(matched, missing) -> float:
    total = len(matched) + len(missing)
    if total == 0:
        return 0.0
    return round((len(matched) / total) * 100, 2)

# ===================== INPUT =====================
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=220)

# ===================== PIPELINE =====================
if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    with st.expander("📄 Resume Text Preview"):
        st.write(resume_text[:1200] + "...")

    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    semantic_score, features, tfidf_matrix = semantic_similarity(
        clean_resume, clean_jd
    )

    matched, missing = keyword_gap_analysis(features, tfidf_matrix)

    experience_score = experience_signal_score(clean_resume)
    skill_score = skill_coverage_score(matched, missing)

    final_score = round(
        (0.5 * semantic_score) +
        (0.3 * skill_score) +
        (0.2 * experience_score),
        2
    )

    # ===================== RESULTS =====================
    st.subheader("📊 ATS Evaluation Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Semantic Match", f"{semantic_score}%")
    col2.metric("Skill Coverage", f"{skill_score}%")
    col3.metric("Experience Signal", f"{experience_score}%")

    st.subheader("🏆 Final ATS Score")
    st.metric("Overall Resume Score", f"{final_score}%")
    st.progress(int(final_score))

    # ===================== DECISION =====================
    st.subheader("📌 Recruiter Screening Outcome")

    if final_score >= 75:
        st.success(
            "Strong candidate. Resume is highly aligned and would likely advance to recruiter review."
        )
    elif final_score >= 60:
        st.warning(
            "Moderate alignment. Resume may pass ATS but rank lower than stronger candidates."
        )
    else:
        st.error(
            "Low alignment. Resume is likely to be filtered out during automated screening."
        )

    # ===================== KEYWORDS =====================
    st.subheader("✅ Strongly Matched Keywords")
    if not matched.empty:
        st.dataframe(matched.head(10)[["Keyword"]])
    else:
        st.write("No strong keyword overlap detected.")

    st.subheader("❌ High-Impact Missing Keywords")
    if not missing.empty:
        st.dataframe(missing.head(10)[["Keyword"]])
    else:
        st.success("No critical keyword gaps detected.")

    # ===================== INSIGHTS =====================
    st.subheader("🧠 Resume Optimization Insights")

    insights = []

    if semantic_score < 60:
        insights.append(
            "Resume language does not sufficiently mirror job description terminology. ATS systems favor semantic similarity."
        )

    if skill_score < 65:
        insights.append(
            "Several role-critical skills are missing or underrepresented and may act as ATS filters."
        )

    if experience_score < 50:
        insights.append(
            "Resume lacks strong action-oriented experience language (projects, impact, outcomes)."
        )

    if insights:
        for tip in insights:
            st.info("💡 " + tip)
    else:
        st.success(
            "Resume content is well-optimized for automated screening systems."
        )

    st.caption(
        "Simulates real-world ATS screening using explainable NLP-based scoring | Portfolio-grade project"
    )
