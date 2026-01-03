import streamlit as st
import fitz
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI-Powered Resume Analyzer")
st.caption("ATS-style resume screening using NLP & ML")

# ---------------- NLP SETUP ----------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def compute_similarity(resume, jd):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000
    )
    tfidf = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(score * 100, 2), vectorizer.get_feature_names_out(), tfidf.toarray()

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

# ---------------- INPUT ----------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=220)

# ---------------- PROCESS ----------------
if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    with st.expander("📄 Resume Text Preview"):
        st.write(resume_text[:1200] + "...")

    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    relevance_score, features, tfidf_matrix = compute_similarity(clean_resume, clean_jd)

    # ---------------- SCORING ----------------
    st.subheader("📊 Overall Resume Match Score")
    st.metric("ATS Relevance Score", f"{relevance_score}%")
    st.progress(int(relevance_score))

    if relevance_score >= 75:
        st.success("Strong alignment. Resume is highly relevant for this role.")
    elif relevance_score >= 55:
        st.warning("Moderate alignment. Targeted optimization recommended.")
    else:
        st.error("Low alignment. Resume likely filtered out by ATS.")

    # ---------------- KEYWORD ANALYSIS ----------------
    matched, missing = keyword_gap_analysis(features, tfidf_matrix)

    st.subheader("✅ Strongly Matched Keywords")
    if not matched.empty:
        st.dataframe(matched.head(10)[["Keyword"]])
    else:
        st.write("No strong overlaps detected.")

    st.subheader("❌ High-Impact Missing Keywords")
    if not missing.empty:
        st.dataframe(missing.head(10)[["Keyword"]])
    else:
        st.success("No critical keyword gaps identified.")

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Recruiter-Focused Resume Insights")

    insights = []

    if relevance_score < 60:
        insights.append(
            "Resume language does not sufficiently mirror the job description. ATS systems favor semantic similarity."
        )

    if len(missing) > 8:
        insights.append(
            "Several high-priority role-specific skills are missing. These are likely used as ATS filters."
        )

    if "experience" not in clean_resume:
        insights.append(
            "Resume lacks strong experience-oriented phrasing (projects, impact, outcomes)."
        )

    if insights:
        for tip in insights:
            st.info("💡 " + tip)
    else:
        st.success("Resume content is well-optimized for automated screening systems.")

    st.caption(
        "Designed to simulate real-world ATS resume screening using NLP, TF-IDF, and cosine similarity"
    )
