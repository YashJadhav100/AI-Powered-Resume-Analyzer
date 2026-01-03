# 🧠 AI-Powered Resume Analyzer

### *ATS-Style Resume Intelligence & Skill Gap Analysis System*

An **industry-inspired, ATS-style resume screening system** that evaluates how well a candidate’s resume aligns with a job description using **Natural Language Processing (NLP)** and **explainable scoring logic**.

This project simulates **real-world automated resume screening** used by recruiters and applicant tracking systems by combining **semantic matching**, **skill coverage analysis**, and **experience signal detection**.

## 🚀 Why This Project Matters

Modern hiring pipelines rely heavily on **Applicant Tracking Systems (ATS)** to filter resumes before human review.

This project is designed to:

* Mimic how ATS tools score and rank resumes
* Explain *why* a resume passes or fails screening
* Identify **high-impact skill gaps**
* Provide **recruiter-facing insights**, not just raw scores

It goes beyond a single similarity metric and instead produces a **multi-signal hiring evaluation**.

## 🎯 Key Capabilities

* 📄 **PDF Resume Parsing** using PyMuPDF
* 🧠 **Text Normalization & Cleaning** using NLTK
* 🔍 **Semantic Resume–JD Matching** using TF-IDF & Cosine Similarity
* 🧩 **Skill Gap & Coverage Analysis**
* 🧠 **Experience Signal Detection** (action-oriented language)
* 🏆 **Weighted ATS-Style Scoring System**
* 📌 **Recruiter-Oriented Screening Decisions**
* 💡 **Explainable Resume Optimization Insights**

## 🧮 ATS-Style Scoring Logic

Instead of relying on a single similarity score, the system computes a **composite ATS score** using weighted signals:

```
Final ATS Score =
  50% Semantic Match Score
+ 30% Skill Coverage Score
+ 20% Experience Signal Score
```

### What This Means:

* **Semantic Match** evaluates how closely resume language mirrors the job description
* **Skill Coverage** measures how many role-critical skills are present
* **Experience Signal** detects action-driven, impact-oriented resume language

This mirrors how real ATS systems rank candidates.

## 📊 Output & Insights

The system produces:

* Overall ATS relevance score (%)
* Breakdown of semantic match, skill coverage, and experience signals
* Matched vs missing high-impact keywords
* Recruiter-style screening outcomes:

  * Likely to advance
  * Borderline candidate
  * Likely filtered out
* Actionable, explainable recommendations for resume optimization

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (UI & deployment)
* **scikit-learn** (TF-IDF, Cosine Similarity)
* **NLTK** (stopword removal, preprocessing)
* **PyMuPDF** (PDF resume extraction)
* **Pandas** (analysis & reporting)

## 📂 Project Structure

```
AI-Powered-Resume-Analyzer/
│
├── app.py              # Main ATS-style resume analysis engine
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── Example.png         # Sample UI screenshot
└── .gitignore
```

## 🧪 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YashJadhav100/AI-Powered-Resume-Analyzer.git
cd AI-Powered-Resume-Analyzer
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

## 🌐 Deployment

This project can be deployed for free using **Streamlit Cloud**:

1. Connect your GitHub repository
2. Select `app.py` as the entry point
3. Click **Deploy**

## 🧠 What This Project Demonstrates to Recruiters

* Practical application of NLP in hiring workflows
* Understanding of ATS screening logic
* Explainable AI decision-making
* Production-aware engineering
* Ability to translate ML outputs into business insights

This is not a toy ML project.
It is a **resume intelligence system** designed with real hiring pipelines in mind.

## 🔮 Future Enhancements

* Sentence-BERT or SBERT for deeper semantic matching
* Section-wise scoring (Skills vs Experience vs Projects)
* Resume rewrite suggestions powered by LLMs
* Skill importance weighting by role seniority
* Visual analytics dashboards

## ✍️ Author
**Yash Jadhav**
Master’s in Computer Science – Syracuse University
