from flask import Flask, render_template, request, redirect, url_for, session
import io
import re

import PyPDF2

# Optional dependency for better text extraction.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Optional OCR dependencies for scanned/image PDFs.
try:
    import pytesseract
    from pdf2image import convert_from_bytes
except ImportError:
    pytesseract = None
    convert_from_bytes = None

app = Flask(__name__)
app.secret_key = "smart_resume_analyzer_secret_key"

# Simple in-memory auth store for sign-up/login.
# Key: email, Value: password
USER_STORE = {}

# Basic email pattern used for server-side validation.
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Expanded skill catalog used to detect resume and job-description skills.
SKILL_CATALOG = {
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b"],
    "SQL": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bpostgres\b"],
    "HTML": [r"\bhtml\b", r"\bhtml5\b"],
    "CSS": [r"\bcss\b", r"\bcss3\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b", r"\becmascript\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    "React": [r"\breact\b", r"\breactjs\b"],
    "Node.js": [r"\bnode\.?js\b", r"\bnodejs\b"],
    "Git": [r"\bgit\b", r"\bgithub\b"],
    "Docker": [r"\bdocker\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
}

# Learning suggestions used for personalized roadmap generation.
LEARNING_RESOURCES = {
    "Python": "Python for Everybody (Coursera) + practice with scripting mini-projects.",
    "Java": "Java Programming and Software Engineering Fundamentals (Coursera).",
    "SQL": "SQL for Data Science (Coursera) + practice joins/subqueries on sample DBs.",
    "HTML": "MDN HTML Guide + build semantic multi-page layouts.",
    "CSS": "MDN CSS Guide + practice Flexbox/Grid responsive pages.",
    "JavaScript": "JavaScript.info or freeCodeCamp JavaScript Algorithms and Data Structures.",
    "Machine Learning": "Andrew Ng Machine Learning Specialization + scikit-learn projects.",
    "Flask": "Flask Official Tutorial + build CRUD apps with forms and templates.",
    "Django": "Django Official Tutorial + create apps with models/admin/auth.",
    "React": "React Official Docs + build state-driven component projects.",
    "Node.js": "Node.js Official Docs + build REST APIs with Express.",
    "Git": "Atlassian Git Tutorials + daily branch/merge workflow practice.",
    "Docker": "Docker Getting Started + containerize Python web applications.",
    "AWS": "AWS Cloud Practitioner Essentials + deploy a simple app on EC2.",
    "Data Analysis": "Google Data Analytics (Coursera) + dashboard/reporting practice.",
    "Pandas": "Kaggle Pandas Micro-Course + data cleaning exercises.",
    "NumPy": "NumPy Official User Guide + vectorized numerical problem solving.",
}

ROLE_QUESTION_BANK = {
    "Data Scientist": [
        "How do you handle overfitting and underfitting in a model?",
        "Which evaluation metric would you pick for imbalanced classification and why?",
        "How do you approach feature engineering in a new dataset?",
        "How would you explain model performance to non-technical stakeholders?",
    ],
    "Machine Learning Engineer": [
        "How would you deploy and monitor a machine learning model in production?",
        "How do you detect and handle data drift?",
        "How would you optimize inference latency for real-time predictions?",
        "How do you design a reproducible ML pipeline?",
    ],
    "Backend Developer": [
        "How do you design secure and versioned REST APIs?",
        "How do you optimize slow database queries in production?",
        "How do you implement authentication and authorization?",
        "How do you ensure reliability and observability in backend services?",
    ],
    "Frontend Developer": [
        "How do you optimize page performance in a JavaScript-heavy application?",
        "How do you structure reusable components and manage state?",
        "How do you ensure accessibility and responsive behavior across devices?",
        "How do you debug browser-specific UI issues?",
    ],
    "Full Stack Developer": [
        "How do you coordinate API contracts between frontend and backend teams?",
        "How do you secure data flow across the full stack?",
        "How do you design end-to-end tests for critical user journeys?",
        "How do you choose technologies for a full stack project?",
    ],
    "Software Engineer": [
        "How do you break down complex features into testable tasks?",
        "How do you debug production issues with limited logs?",
        "How do you balance technical debt with delivery speed?",
        "How do you keep code maintainable in a collaborative team?",
    ],
}


def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF using multiple strategies, including OCR fallback for scanned PDFs."""
    text_chunks = []

    # Strategy 1: pdfplumber extraction (often best for structured text PDFs).
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_chunks.append(page_text)
        except Exception:
            pass

    # Strategy 2: PyPDF2 extraction as secondary parser.
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
    except Exception:
        pass

    combined_text = "\n".join(text_chunks).strip()

    # Strategy 3: OCR fallback for scanned/image-based PDFs.
    if not combined_text and pytesseract is not None and convert_from_bytes is not None:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=220)
            ocr_text = []
            for image in images:
                extracted = pytesseract.image_to_string(image) or ""
                if extracted.strip():
                    ocr_text.append(extracted)
            combined_text = "\n".join(ocr_text).strip()
        except Exception:
            pass

    return combined_text


def identify_skills(text):
    """Return detected skills from the provided text using regex patterns."""
    normalized_text = text.lower()
    found_skills = []

    for skill, patterns in SKILL_CATALOG.items():
        if any(re.search(pattern, normalized_text) for pattern in patterns):
            found_skills.append(skill)

    return sorted(found_skills)


def calculate_match_score(resume_skills, required_skills):
    """Calculate matched skills, missing skills, and resume match percentage."""
    resume_set = set(resume_skills)
    required_set = set(required_skills)

    matched_skills = sorted(resume_set.intersection(required_set))
    missing_skills = sorted(required_set.difference(resume_set))

    if not required_set:
        score = 0.0
    else:
        score = (len(matched_skills) / len(required_set)) * 100

    return matched_skills, missing_skills, round(score, 2)


def generate_weekwise_roadmap(missing_skills):
    """Generate a dynamic week-wise learning roadmap from missing skills."""
    roadmap = []

    # Guidance rotates to create varied weekly actions.
    week_guidance = [
        "Start with fundamentals and key terminology.",
        "Practice with small hands-on exercises and examples.",
        "Build an intermediate mini-project and review best practices.",
    ]

    for index, skill in enumerate(missing_skills):
        week_number = index + 1
        guidance = week_guidance[index % len(week_guidance)]
        resource = LEARNING_RESOURCES.get(
            skill,
            "Study core concepts, complete beginner tutorials, and build a small project.",
        )

        roadmap.append(
            {
                "week": f"Week {week_number}",
                "skill": skill,
                "guidance": guidance,
                "resource": resource,
            }
        )

    return roadmap


def detect_role(job_description):
    """Infer the job role from keywords in the job description."""
    jd = job_description.lower()

    if "data scientist" in jd:
        return "Data Scientist"
    if "machine learning engineer" in jd or "ml engineer" in jd:
        return "Machine Learning Engineer"
    if "backend" in jd or "back-end" in jd:
        return "Backend Developer"
    if "frontend" in jd or "front-end" in jd:
        return "Frontend Developer"
    if "full stack" in jd or "full-stack" in jd:
        return "Full Stack Developer"

    return "Software Engineer"


def generate_interview_questions(job_description, missing_skills, role, limit=8):
    """Generate dynamic role-based interview questions using role and missing skills."""
    questions = list(ROLE_QUESTION_BANK.get(role, ROLE_QUESTION_BANK["Software Engineer"]))

    # Add targeted questions for missing skills to personalize preparation.
    for skill in missing_skills[:4]:
        questions.append(f"How would you apply {skill} in this role on a real project?")

    # Add contextual question based on common JD expectations.
    jd = job_description.lower()
    if "team" in jd or "collabor" in jd:
        questions.append("Describe how you collaborate with cross-functional teams during delivery.")
    if "deadline" in jd or "fast-paced" in jd:
        questions.append("How do you prioritize tasks when working under tight deadlines?")

    # Deduplicate while preserving order.
    unique_questions = []
    seen = set()
    for question in questions:
        if question not in seen:
            unique_questions.append(question)
            seen.add(question)

    return unique_questions[:limit]


@app.route("/", methods=["GET", "POST"])
def signin():
    """Render sign-up/sign-in page and handle authentication."""
    if request.method == "POST":
        auth_action = request.form.get("auth_action", "signup")
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template("signin.html", error="Please enter email and password.")

        # Added server-side email format validation.
        if not EMAIL_REGEX.match(email):
            return render_template("signin.html", error="Please enter a valid email address.")

        # Password validation for sign-up/sign-in input.
        if len(password) < 6:
            return render_template("signin.html", error="Password must be at least 6 characters long.")

        if auth_action == "signup":
            if email in USER_STORE:
                return render_template("signin.html", error="Email already registered. Please sign in.")

            USER_STORE[email] = password
            session["user_email"] = email
            return redirect(url_for("index"))

        # Sign-in flow.
        if email not in USER_STORE or USER_STORE[email] != password:
            return render_template("signin.html", error="Invalid email or password.")

        session["user_email"] = email
        return redirect(url_for("index"))

    return render_template("signin.html")


@app.route("/analyze", methods=["GET", "POST"])
def index():
    """Render analyzer page and process uploaded resume against job description."""
    if "user_email" not in session:
        return redirect(url_for("signin"))

    if request.method == "POST":
        resume_file = request.files.get("resume")
        job_description = request.form.get("job_desc", "").strip()

        if not resume_file or resume_file.filename == "":
            return render_template(
                "index.html",
                user_identity=session["user_email"],
                error="Please upload a resume PDF.",
            )

        if not resume_file.filename.lower().endswith(".pdf"):
            return render_template(
                "index.html",
                user_identity=session["user_email"],
                error="Only PDF files are supported.",
            )

        if not job_description:
            return render_template(
                "index.html",
                user_identity=session["user_email"],
                error="Please paste a job description.",
            )

        pdf_bytes = resume_file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)

        if not resume_text.strip():
            return render_template(
                "index.html",
                user_identity=session["user_email"],
                error=(
                    "Could not extract text from this PDF. "
                    "If it is scanned, install OCR dependencies: pytesseract + pdf2image + Tesseract OCR."
                ),
            )

        resume_skills = identify_skills(resume_text)
        required_skills = identify_skills(job_description)

        matched_skills, missing_skills, score = calculate_match_score(resume_skills, required_skills)

        # Week-wise roadmap is generated dynamically for each resume's missing skills.
        roadmap = generate_weekwise_roadmap(missing_skills)

        role = detect_role(job_description)
        interview_questions = generate_interview_questions(job_description, missing_skills, role)

        return render_template(
            "result.html",
            user_identity=session["user_email"],
            role=role,
            match_score=score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            roadmap=roadmap,
            interview_questions=interview_questions,
        )

    return render_template("index.html", user_identity=session["user_email"])


@app.route("/logout")
def logout():
    """Clear session and return to sign-in page."""
    session.clear()
    return redirect(url_for("signin"))


if __name__ == "__main__":
    app.run(debug=True)
