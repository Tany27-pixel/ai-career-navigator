from io import BytesIO
import html
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# LOAD CAREER DATASET
# --------------------------------------------------

@st.cache_data
def load_careers():
    df = pd.read_csv("careers.csv")

    required = ["career", "interest", "description", "skills"]
    if not all(column in df.columns for column in required):
        raise ValueError(
            "careers.csv must contain these columns: "
            "career, interest, description, skills."
        )

    df = df.dropna(subset=required).copy()
    return df.drop_duplicates(subset=["career"])


try:
    careers_df = load_careers()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load careers.csv: {error}")
    st.info(
        "Keep careers.csv in the same folder as app.py and check its "
        "column names."
    )
    st.stop()

if careers_df.empty:
    st.error("Your career dataset is empty.")
    st.stop()

# --------------------------------------------------
# BUILD TF-IDF RECOMMENDATION MODEL
# --------------------------------------------------

@st.cache_resource
def build_model(career_texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    career_vectors = vectorizer.fit_transform(career_texts)
    return vectorizer, career_vectors


career_texts = (
    careers_df["career"].astype(str) + " " +
    careers_df["interest"].astype(str) + " " +
    careers_df["description"].astype(str) + " " +
    careers_df["skills"].astype(str).str.replace(";", " ", regex=False)
).tolist()

try:
    vectorizer, career_vectors = build_model(career_texts)
except ValueError:
    st.error("The career dataset has no usable text to learn from.")
    st.stop()

# --------------------------------------------------
# CAREER RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_careers(selected_interests):
    query = " ".join(selected_interests)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, career_vectors).flatten()

    results = careers_df.copy()
    results["similarity"] = scores
    results["direct_match"] = results["interest"].isin(selected_interests)

    return results.sort_values(
        ["direct_match", "similarity"],
        ascending=[False, False],
    )

# --------------------------------------------------
# PERSONALIZED CAREER ROADMAPS
# --------------------------------------------------

CAREER_ROADMAPS = {
    "AI Engineer": [
        ("Foundation", "Learn Python, functions, and data structures."),
        ("Mathematics", "Study linear algebra, probability, and statistics."),
        ("Machine Learning", "Learn scikit-learn and build prediction models."),
        ("Deep Learning", "Explore neural networks and PyTorch fundamentals."),
        ("Projects", "Build an AI chatbot or document-question-answering project."),
        ("Portfolio", "Document your projects on GitHub."),
    ],
    "Machine Learning Engineer": [
        ("Foundation", "Learn Python, NumPy, pandas, and Git."),
        ("Mathematics", "Practice statistics, probability, and linear algebra."),
        ("Machine Learning", "Study regression, classification, and evaluation."),
        ("Model Building", "Practice scikit-learn pipelines and preprocessing."),
        ("Deployment", "Deploy a model with Streamlit or an API."),
        ("Portfolio", "Build and document two end-to-end ML projects."),
    ],
    "NLP Engineer": [
        ("Foundation", "Learn Python, strings, lists, and dictionaries."),
        ("Text Processing", "Explore text cleaning, tokenization, and TF-IDF."),
        ("NLP Basics", "Practice text classification and sentiment analysis."),
        ("Deep Learning", "Learn embeddings and transformer fundamentals."),
        ("Projects", "Build a text classifier or FAQ assistant."),
        ("Portfolio", "Publish your code and evaluation results."),
    ],
    "Data Scientist": [
        ("Foundation", "Learn Python, pandas, and NumPy."),
        ("Statistics", "Practice probability and hypothesis testing."),
        ("Data Analysis", "Clean, explore, and visualize real datasets."),
        ("Machine Learning", "Train and evaluate prediction models."),
        ("Projects", "Complete an analysis and a prediction project."),
        ("Portfolio", "Present your findings on GitHub."),
    ],
    "Data Analyst": [
        ("Foundation", "Learn spreadsheets and basic data concepts."),
        ("SQL", "Practice filtering, joins, grouping, and aggregate functions."),
        ("Python", "Learn pandas for data cleaning and analysis."),
        ("Visualization", "Create charts and dashboards."),
        ("Projects", "Analyze a real dataset and build a dashboard."),
        ("Portfolio", "Explain your insights and recommendations."),
    ],
    "Frontend Developer": [
        ("Foundation", "Learn HTML and semantic page structure."),
        ("Styling", "Practice CSS, layouts, and responsive design."),
        ("JavaScript", "Learn variables, functions, events, and DOM basics."),
        ("Frameworks", "Explore a framework such as React."),
        ("Projects", "Build a responsive interactive website."),
        ("Portfolio", "Publish your projects and source code."),
    ],
    "Backend Developer": [
        ("Foundation", "Learn Python and programming fundamentals."),
        ("Web Basics", "Understand HTTP, requests, responses, and APIs."),
        ("Framework", "Build an application using Flask or FastAPI."),
        ("Databases", "Practice SQL, tables, joins, and database operations."),
        ("Projects", "Create an API with validation and error handling."),
        ("Portfolio", "Document setup instructions and API endpoints."),
    ],
    "Security Analyst": [
        ("Foundation", "Learn operating systems, networking, and Linux."),
        ("Linux", "Practice Linux commands and file permissions."),
        ("Security Basics", "Study common threats, logs, and security principles."),
        ("Monitoring", "Interpret sample logs and security alerts."),
        ("Practice", "Use authorized cybersecurity labs and sample datasets."),
        ("Portfolio", "Document defensive security analysis exercises."),
    ],
    "Software Developer": [
        ("Foundation", "Choose a programming language and learn its fundamentals."),
        ("Problem Solving", "Practice data structures and algorithms."),
        ("Development Tools", "Learn Git, debugging, and testing."),
        ("Applications", "Study APIs, databases, and application design."),
        ("Projects", "Build a complete application."),
        ("Portfolio", "Publish readable code and documentation."),
    ],
    "Cloud Engineer": [
        ("Foundation", "Learn operating systems, networking, and Linux."),
        ("Scripting", "Practice Python or shell scripting."),
        ("Cloud Basics", "Study cloud computing, storage, and networking."),
        ("Deployment", "Deploy a small application in a cloud learning environment."),
        ("Automation", "Explore containers and infrastructure automation."),
        ("Portfolio", "Document a small, secure cloud deployment project."),
    ],
}


def get_roadmap(career_name):
    return CAREER_ROADMAPS.get(
        career_name,
        [
            ("Foundation", "Learn the core concepts for this career."),
            ("Core Skills", "Practice the main tools and technologies."),
            ("Projects", "Build a practical project."),
            ("Portfolio", "Document your work and learning."),
        ],
    )

# --------------------------------------------------
# PDF REPORT FUNCTIONS
# --------------------------------------------------

def safe_text(value):
    return html.escape(str(value)).replace("\n", "<br/>")


def create_career_report(name, education, interests, results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#263c68"),
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=16,
    ))

    story = [
        Paragraph("AI Career Navigator", styles["ReportTitle"]),
        Paragraph("Your Personalized Career Report", styles["Heading2"]),
        Spacer(1, 12),
    ]

    profile = [
        [Paragraph("<b>Name</b>", styles["BodyText"]), Paragraph(safe_text(name), styles["BodyText"])],
        [Paragraph("<b>Education</b>", styles["BodyText"]), Paragraph(safe_text(education), styles["BodyText"])],
        [Paragraph("<b>Interests</b>", styles["BodyText"]), Paragraph(safe_text(", ".join(interests)), styles["BodyText"])],
    ]

    profile_table = Table(profile, colWidths=[100, 380])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8edf7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.extend([
        profile_table,
        Spacer(1, 18),
        Paragraph("Top 5 Career Matches", styles["Heading1"]),
    ])

    for number, (_, career) in enumerate(results.head(5).iterrows(), start=1):
        story.extend([
            Spacer(1, 10),
            Paragraph(f"{number}. {safe_text(career['career'])}", styles["Heading2"]),
            Paragraph(safe_text(career["description"]), styles["BodyText"]),
            Spacer(1, 5),
            Paragraph(
                "<b>Skills to learn:</b> " +
                safe_text(str(career["skills"]).replace(";", ", ")),
                styles["BodyText"],
            ),
            Paragraph(
                f"<b>Text similarity score:</b> {float(career['similarity']):.2f}",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph("Learning Roadmap", styles["Heading3"]),
        ])

        for step_number, (phase, task) in enumerate(get_roadmap(career["career"]), start=1):
            story.append(Paragraph(
                f"<b>Step {step_number}: {safe_text(phase)}</b> — {safe_text(task)}",
                styles["BodyText"],
            ))
        story.append(Spacer(1, 10))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(
            letter[0] / 2, 25,
            f"AI Career Navigator | Page {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return buffer.getvalue()

# --------------------------------------------------
# PREMIUM WEBSITE THEME (ONE CSS BLOCK)
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #080d1a 0%, #111c33 55%, #1b1640 100%);
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #0d1425;
        border-right: 1px solid #293750;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 26, 0.85);
    }

    .hero {
        padding: 32px;
        border: 1px solid #4654a1;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #4c1d95, #312e81);
        margin-bottom: 28px;
    }

    [data-testid="stMetric"] {
        background: #151f32;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 16px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(21, 31, 50, 0.65);
        border: 1px solid #334155;
        border-radius: 16px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div.stLinkButton > a {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #6475dc;
    }

    hr {
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎯 AI Career Navigator</h1>
    <h3>Discover a career that matches your skills!</h3>
    <p>Explore career matches, skills and personalized learning paths.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# USER INFORMATION FORM
# --------------------------------------------------

st.header("👩‍🎓 Tell us about yourself")

with st.form("career_form"):
    name = st.text_input("What is your name?")

    education = st.selectbox(
        "What are you studying?",
        [
            "Select your education",
            "Computer Science",
            "Data Science",
            "Information Technology",
            "Engineering - Other",
            "BCA / MCA",
            "Other",
        ],
    )

    interests = st.multiselect(
        "Which areas interest you?",
        [
            "Artificial Intelligence",
            "Data Science",
            "Web Development",
            "Cybersecurity",
            "Software Engineering",
            "Cloud Computing",
        ],
    )

    submitted = st.form_submit_button(
        "✨ Find My Career Matches",
        use_container_width=True,
    )

# --------------------------------------------------
# DISPLAY CAREER MATCHES, ANALYTICS AND ROADMAPS
# --------------------------------------------------

if submitted:
    if not name.strip():
        st.warning("Please enter your name.")
    elif education == "Select your education":
        st.warning("Please select your education.")
    elif not interests:
        st.warning("Please select at least one interest.")
    else:
        st.success(f"Welcome, {name.strip()}! 🎉")
        st.write("**Education:**", education)
        st.write("**Interests:**", ", ".join(interests))

        results = recommend_careers(interests)

        # DASHBOARD SUMMARY
        st.divider()
        st.subheader("✨ Your Career Dashboard")
        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.metric("Career Matches", len(results.head(5)))
        with metric2:
            st.metric("Interests Selected", len(interests))
        with metric3:
            st.metric("Careers in Dataset", len(careers_df))

        # CAREER MATCH ANALYTICS
        st.divider()
        st.subheader("📊 Career Match Analytics")
        top_five = results.head(5).copy()
        chart_data = top_five.set_index("career")["similarity"]

        st.write("Compare text similarity across your top 5 careers.")
        st.bar_chart(chart_data, y_label="Similarity Score", x_label="Career")
        st.caption(
            "Scores measure text similarity, not your probability of getting "
            "a job or succeeding in that career."
        )

        # CAREER MATCH CARDS
        st.divider()
        st.header("🚀 Your Career Matches")
        st.caption(
            "Rankings use text similarity and direct interest matches. "
            "They are suggestions, not probabilities of career success."
        )

        for _, career in results.head(5).iterrows():
            with st.container(border=True):
                st.subheader("💼 " + career["career"])
                st.write(career["description"])
                st.write("**Skills to learn:**", str(career["skills"]).replace(";", ", "))

                score = float(career["similarity"])
                st.progress(max(0.0, min(score, 1.0)))
                st.caption(f"Text similarity score: {score:.2f}")

                with st.expander(f"🗺️ View {career['career']} Roadmap"):
                    st.write("Follow these steps at your own pace.")
                    for step_number, (phase, task) in enumerate(get_roadmap(career["career"]), start=1):
                        st.markdown(f"**Step {step_number}: {phase}**")
                        st.write(task)
                        st.checkbox(
                            f"Mark {phase} as completed",
                            key=f"roadmap_{career['career']}_{step_number}",
                        )

        # SKILL-GAP ANALYZER
        st.divider()
        st.header("🧠 Skill-Gap Analyzer")
        st.write(
            "Select the skills you already know to discover what you can "
            "learn next for your chosen career."
        )

        all_skills = sorted({
            skill.strip()
            for skills_text in careers_df["skills"]
            for skill in str(skills_text).split(";")
            if skill.strip()
        })

        known_skills = st.multiselect(
            "✅ Which skills do you already know?",
            options=all_skills,
            key="known_skills",
        )

        career_options = results.head(5)["career"].tolist()
        selected_career = st.selectbox(
            "🎯 Choose a career to analyze",
            options=career_options,
            key="skill_gap_career",
        )

        selected_row = results[results["career"] == selected_career].iloc[0]
        required_skills = [
            skill.strip()
            for skill in str(selected_row["skills"]).split(";")
            if skill.strip()
        ]

        known_lower = {skill.lower() for skill in known_skills}
        matched_skills = [skill for skill in required_skills if skill.lower() in known_lower]
        missing_skills = [skill for skill in required_skills if skill.lower() not in known_lower]

        total_skills = len(required_skills)
        readiness = len(matched_skills) / total_skills if total_skills else 0

        st.subheader(f"📈 Skill Coverage: {readiness:.0%}")
        st.progress(readiness)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Skills You Know")
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"• {skill}")
            else:
                st.info("No matching required skills selected yet.")

        with col2:
            st.markdown("### 📚 Skills to Learn")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"• {skill}")
            else:
                st.success("You've selected all listed skills for this career!")

        if missing_skills:
            st.info(
                "💡 Start with one missing skill, practice it, and build a "
                "small project using it."
            )

        st.caption(
            "Skill coverage compares your selections with the skills listed "
            "in this project's dataset. It is not a job-readiness assessment "
            "or a prediction of employment."
        )

        # PERSONALIZED PROJECT IDEAS
        st.divider()
        st.header("🛠️ Project Ideas for Your Career")

        project_ideas = {
            "AI Engineer": [
                "Build a simple AI question-answering assistant.",
                "Create an image classification demo.",
                "Build a recommendation system for books or movies.",
            ],
            "Machine Learning Engineer": [
                "Build a house-price prediction model using sample data.",
                "Compare classification models on a public dataset.",
                "Create a reusable machine-learning pipeline.",
            ],
            "NLP Engineer": [
                "Create a sentiment analysis tool.",
                "Build a text classifier for sample reviews.",
                "Make a frequently asked questions chatbot.",
            ],
            "Data Scientist": [
                "Explore a public dataset and explain key trends.",
                "Build a prediction model and evaluate its performance.",
                "Create a notebook with a complete data science workflow.",
            ],
            "Data Analyst": [
                "Create a sales dashboard from sample data.",
                "Analyze customer feedback and summarize trends.",
                "Write SQL queries to answer business questions.",
            ],
            "Frontend Developer": [
                "Build a responsive personal portfolio website.",
                "Create an interactive to-do list.",
                "Design a dashboard interface using HTML and CSS.",
            ],
            "Backend Developer": [
                "Create a task-management API.",
                "Build a small API connected to a database.",
                "Add validation and clear error messages to an API.",
            ],
            "Security Analyst": [
                "Analyze sample security logs.",
                "Create a checklist for securing a demo application.",
                "Build a simple log-filtering and alert-reporting tool.",
            ],
            "Software Developer": [
                "Build a command-line expense tracker.",
                "Create a library or inventory management app.",
                "Write unit tests for a small application.",
            ],
            "Cloud Engineer": [
                "Document a basic cloud architecture diagram.",
                "Deploy a small demo app using a cloud learning environment.",
                "Create a checklist for cloud access and security settings.",
            ],
        }

        project_career = st.selectbox(
            "Choose a career for project ideas",
            options=career_options,
            key="project_ideas_career",
        )
        for idea in project_ideas.get(project_career, ["Build a small project related to this career."]):
            st.markdown(f"- {idea}")

        # PERSONALIZED LEARNING RESOURCES
        st.divider()
        st.header("📚 Personalized Learning Resources")
        learning_resources = {
            "Python": ("Python Tutorial", "https://docs.python.org/3/tutorial/"),
            "Machine Learning": ("Google Machine Learning Crash Course", "https://developers.google.com/machine-learning/crash-course"),
            "Scikit-learn": ("Scikit-learn User Guide", "https://scikit-learn.org/stable/user_guide.html"),
            "Statistics": ("Khan Academy Statistics", "https://www.khanacademy.org/math/statistics-probability"),
            "SQL": ("SQLBolt Interactive Lessons", "https://sqlbolt.com/"),
            "Data Visualization": ("Plotly Python Documentation", "https://plotly.com/python/"),
            "NLP": ("Hugging Face NLP Course", "https://huggingface.co/learn/nlp-course/"),
            "Deep Learning": ("PyTorch Tutorials", "https://pytorch.org/tutorials/"),
            "Mathematics": ("Khan Academy Mathematics", "https://www.khanacademy.org/math"),
            "JavaScript": ("MDN JavaScript Guide", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
            "HTML": ("MDN HTML Guide", "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Structuring_content"),
            "CSS": ("MDN CSS Guide", "https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Styling_basics"),
            "Cloud Computing": ("AWS Skill Builder", "https://skillbuilder.aws/"),
            "Cybersecurity": ("Cisco Networking Academy", "https://www.netacad.com/"),
            "Networking": ("Cisco Networking Academy", "https://www.netacad.com/"),
        }

        resource_skills = st.multiselect(
            "🎯 Select skills you want to learn",
            options=sorted(learning_resources.keys()),
            default=["Python"],
            key="learning_resource_skills",
        )

        if resource_skills:
            for skill in resource_skills:
                resource_name, resource_url = learning_resources[skill]
                with st.container(border=True):
                    st.markdown(f"**📖 {skill}**")
                    st.write(resource_name)
                    st.link_button("Open Learning Resource ↗", resource_url)
        else:
            st.info("Select at least one skill to see learning resources.")

        # CAREER APTITUDE QUIZ
        st.divider()
        st.header("🧠 Career Aptitude Quiz")
        st.write(
            "Answer these questions about your interests and preferred "
            "activities to explore suitable career paths."
        )

        with st.form("career_aptitude_quiz"):
            q1 = st.radio(
                "1. Which activity sounds most interesting?",
                [
                    "Building AI models and intelligent applications",
                    "Finding patterns in data and making charts",
                    "Creating websites and applications",
                    "Protecting systems and investigating security issues",
                ],
                index=None,
            )
            q2 = st.radio(
                "2. Which task would you enjoy most?",
                [
                    "Training a model to make predictions",
                    "Analyzing data to discover useful insights",
                    "Designing and building an interactive application",
                    "Finding weaknesses and improving system security",
                ],
                index=None,
            )
            q3 = st.radio(
                "3. Which topic would you most like to learn?",
                [
                    "Artificial intelligence and machine learning",
                    "Statistics, SQL, and data visualization",
                    "Programming, web development, and software design",
                    "Networking, cybersecurity, and cloud systems",
                ],
                index=None,
            )
            quiz_submitted = st.form_submit_button(
                "✨ Discover My Career Matches",
                use_container_width=True,
            )

        if quiz_submitted:
            if not all([q1, q2, q3]):
                st.warning("Please answer all three questions first.")
            else:
                answer_mapping = {
                    "Building AI models and intelligent applications": "AI & Machine Learning",
                    "Finding patterns in data and making charts": "Data Analytics & Science",
                    "Creating websites and applications": "Software Development",
                    "Protecting systems and investigating security issues": "Cybersecurity & Cloud",
                    "Training a model to make predictions": "AI & Machine Learning",
                    "Analyzing data to discover useful insights": "Data Analytics & Science",
                    "Designing and building an interactive application": "Software Development",
                    "Finding weaknesses and improving system security": "Cybersecurity & Cloud",
                    "Artificial intelligence and machine learning": "AI & Machine Learning",
                    "Statistics, SQL, and data visualization": "Data Analytics & Science",
                    "Programming, web development, and software design": "Software Development",
                    "Networking, cybersecurity, and cloud systems": "Cybersecurity & Cloud",
                }

                answer_scores = {
                    "AI & Machine Learning": 0,
                    "Data Analytics & Science": 0,
                    "Software Development": 0,
                    "Cybersecurity & Cloud": 0,
                }

                for answer in [q1, q2, q3]:
                    answer_scores[answer_mapping[answer]] += 1

                ranked_categories = sorted(
                    answer_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                st.subheader("🎯 Your Career Quiz Results")
                for category, score in ranked_categories:
                    if score > 0:
                        st.write(f"**{category}** — {score}/3 answers")
                        st.progress(score / 3)

                top_category = ranked_categories[0][0]
                career_suggestions = {
                    "AI & Machine Learning": ["AI Engineer", "Machine Learning Engineer", "NLP Engineer"],
                    "Data Analytics & Science": ["Data Scientist", "Data Analyst"],
                    "Software Development": ["Software Developer", "Frontend Developer", "Backend Developer"],
                    "Cybersecurity & Cloud": ["Security Analyst", "Cloud Engineer"],
                }

                st.markdown("### 💼 Careers to Explore")
                for career in career_suggestions[top_category]:
                    st.write(f"• {career}")

                st.caption(
                    "This introductory quiz reflects your selected preferences. "
                    "It is for career exploration only, not a validated aptitude "
                    "test or a guarantee of career suitability."
                )

        # PDF DOWNLOAD
        st.divider()
        st.subheader("📄 Download Your Career Report")
        try:
            pdf_bytes = create_career_report(
                name.strip(),
                education,
                interests,
                results,
            )
            safe_filename = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "user"

            st.download_button(
                label="⬇️ Download Career Report (PDF)",
                data=pdf_bytes,
                file_name=f"career_report_{safe_filename}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Could not create the PDF report: {error}")

# --------------------------------------------------
# VIEW CAREER DATASET
# --------------------------------------------------

with st.expander("📊 View Career Dataset"):
    st.dataframe(careers_df, use_container_width=True)
