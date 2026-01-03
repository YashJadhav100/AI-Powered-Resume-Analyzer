import streamlit as st
import fitz  # PyMuPDF
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="centered"
)

st.title("📄 AI-Powered Resume Analyzer")
st.caption("Smart resume-to-job matching using NLP & ML")

# ------------------ NLP SETUP ------------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()

# ------------------ FUNCTIONS ------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_similarity(resume, jd):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000
    )
    tfidf = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(score * 100, 2), vectorizer.get_feature_names_out(), tfidf.toarray()

def keyword_analysis(feature_names, tfidf_matrix):
    resume_weights = tfidf_matrix[0]
    jd_weights = tfidf_matrix[1]

    keywords = pd.DataFrame({
        "keyword": feature_names,
        "resume_score": resume_weights,
        "jd_score": jd_weights
    })

    keywords["gap"] = keywords["jd_score"] - keywords["resume_score"]

    matched = keywords[(keywords.resume_score > 0) & (keywords.jd_score > 0)]
    missing = keywords[(keywords.jd_score > 0) & (keywords.resume_score == 0)]

    return matched.sort_values("jd_score", ascending=False), \
           missing.sort_values("jd_score", ascending=False)

# ------------------ INPUT ------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=200)

# ------------------ PROCESS ------------------
if uploaded_file and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    with st.expander("📄 Resume Preview"):
        st.write(resume_text[:1500] + "...")

    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    relevance_score, features, tfidf_matrix = get_similarity(clean_resume, clean_jd)

    # ------------------ SCORE ------------------
    st.subheader("📊 Resume Match Score")
    st.metric("Relevance Score", f"{relevance_score}%")
    st.progress(int(relevance_score))

    if relevance_score >= 75:
        st.success("Strong match. Resume is well-aligned with the role.")
    elif relevance_score >= 50:
        st.warning("Moderate match. Some optimization recommended.")
    else:
        st.error("Low match. Resume needs targeted improvements.")

    # ------------------ KEYWORDS ------------------
    matched, missing = keyword_analysis(features, tfidf_matrix)

    st.subheader("✅ Strongly Matched Skills")
    if not matched.empty:
        st.dataframe(matched.head(10)[["keyword"]])
    else:
        st.write("No strong keyword overlap found.")

    st.subheader("❌ High-Impact Missing Skills")
    if not missing.empty:
        st.dataframe(missing.head(10)[["keyword"]])
    else:
        st.success("No critical skills missing!")

    # ------------------ INSIGHTS ------------------
    st.subheader("🧠 Resume Improvement Insights")

    insights = []
    if len(missing) > 8:
        insights.append("Your resume lacks several high-impact role-specific skills.")
    if relevance_score < 60:
        insights.append("Consider tailoring your experience bullets to mirror the job description language.")
    if "experience" not in clean_resume:
        insights.append("Resume may be missing explicit experience-based phrasing.")

    if insights:
        for tip in insights:
            st.info("💡 " + tip)
    else:
        st.success("Your resume is well-optimized for this role.")

    st.caption("Built with NLP, TF-IDF, and cosine similarity | Interview-grade project")
