from flask import Flask, render_template, request
import pandas as pd
import nltk
import string
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# optional Gemini import
try:
    from gemini_helper import analyze_resume_with_gemini
except:
    analyze_resume_with_gemini = None

nltk.download('punkt')

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Load Skills Dictionary (DMA)
# -----------------------------
with open("skills.txt", "r") as f:
    SKILLS = [line.strip().lower() for line in f.readlines()]

# -----------------------------
# Load Course Dataset
# -----------------------------
courses_df = pd.read_csv("courses.csv")

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text.lower()

# -----------------------------
# NLP Preprocessing
# -----------------------------
def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)

# -----------------------------
# Skill Extraction (DMA)
# -----------------------------
def extract_skills(text):

    found_skills = []

    for skill in SKILLS:

        if skill in text:

            found_skills.append(skill)

    return list(set(found_skills))

# -----------------------------
# Resume Weakness Detection
# -----------------------------
def detect_weakness(text):

    weaknesses = []

    if "project" not in text:
        weaknesses.append("No projects mentioned")

    if "internship" not in text:
        weaknesses.append("No internship experience")

    if "skills" not in text:
        weaknesses.append("Skills section not clearly defined")

    if len(text.split()) < 150:
        weaknesses.append("Resume content is too short")

    return weaknesses

# -----------------------------
# Course Recommendation
# -----------------------------
def recommend_courses(missing_skills):

    recommended = []

    for skill in missing_skills:

        course = courses_df[courses_df['skill'].str.lower() == skill]

        if not course.empty:

            recommended.append({
                "name": course.iloc[0]['course_name'],
                "link": course.iloc[0]['course_link']
            })

    return recommended

# -----------------------------
# Generate PDF Report
# -----------------------------
def generate_pdf(result):

    file_path = "uploads/report.pdf"

    c = canvas.Canvas(file_path, pagesize=letter)

    c.drawString(100,750,"AI Job Gap Analysis Report")

    c.drawString(100,720,f"Match Score: {result['score']}%")

    c.drawString(100,700,f"ATS Score: {result['ats_score']}%")

    y = 660

    c.drawString(100,y,"Missing Skills:")
    y -= 20

    for skill in result["gap"]:
        c.drawString(120,y,skill)
        y -= 20

    y -= 20
    c.drawString(100,y,"Resume Weakness:")
    y -= 20

    for w in result["weakness"]:
        c.drawString(120,y,w)
        y -= 20

    c.save()

    return file_path

# -----------------------------
# Main Route
# -----------------------------
@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        pdf_file = request.files["resume_pdf"]

        job_desc = request.form["job_desc"]

        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)

        pdf_file.save(pdf_path)

        # Extract Resume Text
        resume_text = extract_text_from_pdf(pdf_path)

        # NLP Preprocessing
        resume_clean = preprocess(resume_text)

        job_clean = preprocess(job_desc)

        # ML Similarity (TF-IDF)
        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        score = round(similarity * 100, 2)

        # Skill Mining
        resume_skills = extract_skills(resume_clean)

        job_skills = extract_skills(job_clean)

        # Skill Gap
        skill_gap = list(set(job_skills) - set(resume_skills))

        # Resume Weakness
        weaknesses = detect_weakness(resume_clean)

        # Course Recommendation
        courses = recommend_courses(skill_gap)

        # ATS Score
        ats_score = score

        if len(skill_gap) > 5:
            ats_score -= 10
        elif len(skill_gap) > 2:
            ats_score -= 5

        ats_score = max(0, min(100, ats_score))

        # Gemini AI (optional)
        gemini_feedback = ""

        if analyze_resume_with_gemini:
            try:
                gemini_feedback = analyze_resume_with_gemini(
                    resume_text,
                    job_desc,
                    skill_gap
                )
            except:
                gemini_feedback = """
AI Feedback (Offline Mode)

• Improve your resume by adding more projects
• Learn the missing skills listed above
• Add measurable achievements
• Tailor resume for each job description
"""

        result = {
            "score": score,
            "ats_score": ats_score,
            "gap": skill_gap,
            "weakness": weaknesses,
            "courses": courses,
            "ai_feedback": gemini_feedback
        }

        pdf_file = generate_pdf(result)

        result["pdf"] = pdf_file

        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)