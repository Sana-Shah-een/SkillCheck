
import re
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SkillCheck | Career Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM COLOR THEME
# =========================================================

STYLE_FILE = Path(__file__).parent / "style.css"
st.markdown(
    f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD JOB DATA
# =========================================================

DATA_FILE = Path(__file__).parent / "jobs.csv"


@st.cache_data
def load_jobs():
    if not DATA_FILE.exists():
        return pd.DataFrame(
            columns=["job_role", "required_skills"]
        )

    data = pd.read_csv(DATA_FILE)

    if "job_role" not in data.columns:
        raise ValueError(
            "jobs.csv must contain a 'job_role' column."
        )

    if "required_skills" not in data.columns:
        raise ValueError(
            "jobs.csv must contain a 'required_skills' column."
        )

    data["job_role"] = data["job_role"].fillna("").astype(str)
    data["required_skills"] = (
        data["required_skills"].fillna("").astype(str)
    )

    data = data[data["job_role"].str.strip() != ""]

    return data


try:
    jobs = load_jobs()
except Exception as error:
    st.error(f"Could not load jobs.csv: {error}")
    st.stop()

if jobs.empty:
    st.error(
        "No job roles found. Add job_role and required_skills "
        "columns to your jobs.csv file."
    )
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_skill(skill):
    """Normalize a skill for case-insensitive matching."""
    return re.sub(r"\s+", " ", str(skill).strip().lower())


def parse_skills(text):
    """Parse comma-separated skills and remove duplicates."""
    seen = set()
    result = []

    for skill in text.split(","):
        skill = skill.strip()
        normalized = normalize_skill(skill)

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(skill)

    return result


def get_skill_match(user_skills, required_skills):
    """Find exact normalized matches and missing skills."""
    user_set = {
        normalize_skill(skill)
        for skill in user_skills
    }

    matched = [
        skill for skill in required_skills
        if normalize_skill(skill) in user_set
    ]

    missing = [
        skill for skill in required_skills
        if normalize_skill(skill) not in user_set
    ]

    return matched, missing


def calculate_similarity(user_skills, required_skills):
    """Calculate TF-IDF cosine similarity between skill lists."""
    if not user_skills or not required_skills:
        return 0.0

    documents = [
        " ".join(user_skills),
        " ".join(required_skills),
    ]

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w+#.]+\b"
    )

    try:
        vectors = vectorizer.fit_transform(documents)

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]

        return round(float(score) * 100, 2)

    except ValueError:
        return 0.0


def analyze_role(user_skills, role):
    """Generate the analysis for one selected job role."""
    selected = jobs[jobs["job_role"] == role].iloc[0]

    required = parse_skills(selected["required_skills"])
    matched, missing = get_skill_match(user_skills, required)

    total = len(required)
    coverage = round(len(matched) / total * 100, 1) if total else 0

    similarity = calculate_similarity(user_skills, required)

    return {
        "job_role": role,
        "user_skills": user_skills,
        "required_skills": required,
        "matched": matched,
        "missing": missing,
        "coverage": coverage,
        "similarity": similarity,
    }


def make_result_id(role, user_skills):
    """Give each unique analysis its own checklist."""
    content = role + "|" + "|".join(
        sorted(normalize_skill(skill) for skill in user_skills)
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# =========================================================
# SESSION STATE
# =========================================================

if "skillcheck_result" not in st.session_state:
    st.session_state["skillcheck_result"] = None

if "skillcheck_name" not in st.session_state:
    st.session_state["skillcheck_name"] = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🎯 SkillCheck")
    st.caption("CAREER INTELLIGENCE")

    st.markdown("---")

    st.subheader("How it works")

    st.markdown("""
    **01** — Add your current skills

    **02** — Choose a target career

    **03** — Discover your skill gaps

    **04** — Track your learning progress
    """)

    st.markdown("---")

    st.subheader("Analysis guide")

    st.markdown("""
    - 🟢 **Matched:** skills you already listed
    - 🟡 **Missing:** skills to learn or practice
    - 🔵 **Coverage:** percentage of required skills matched
    """)

    st.markdown("---")

    st.caption(
        "Results are based on your sample job dataset. "
        "They do not predict hiring outcomes."
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">Your career • Your next step</div>
    <div class="hero-title">🎯 SkillCheck</div>
    <div class="hero-subtitle">
        Turn your existing skills into a focused learning roadmap.
        Discover what you know, what is missing, and where to grow next.
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "Tip: enter skills you genuinely know. Skill coverage is based "
    "on exact skill-name matching, so use the terms in your job dataset."
)


# =========================================================
# USER INPUT
# =========================================================

st.header("Build your skill profile")

col1, col2 = st.columns(2)

with col1:
    user_name = st.text_input(
        "Your name (optional)",
        value=st.session_state["skillcheck_name"],
        placeholder="Enter your name",
        key="profile_name_input",
    )

with col2:
    job_role = st.selectbox(
        "Choose your target job role",
        jobs["job_role"].tolist(),
        key="target_job_role",
    )

user_skills_text = st.text_area(
    "Your current skills",
    placeholder="Python, SQL, OOP, Git, Pandas",
    height=120,
    help="Separate each skill with a comma.",
    key="skills_input",
)

analyze = st.button(
    "✨ Analyze My Skills",
    type="primary",
    use_container_width=True,
)

if analyze:
    user_skills = parse_skills(user_skills_text)

    if not user_skills:
        st.warning("Please enter at least one skill before analyzing.")
    else:
        result = analyze_role(user_skills, job_role)
        result["result_id"] = make_result_id(job_role, user_skills)

        st.session_state["skillcheck_result"] = result
        st.session_state["skillcheck_name"] = user_name.strip()


# =========================================================
# DISPLAY SAVED ANALYSIS
# This section runs again when a checkbox is clicked.
# The saved analysis remains available in session_state.
# =========================================================

result = st.session_state["skillcheck_result"]

if result is not None:

    saved_name = st.session_state["skillcheck_name"]
    role = result["job_role"]
    user_skills = result["user_skills"]
    required_skills = result["required_skills"]
    matched = result["matched"]
    missing = result["missing"]
    coverage = result["coverage"]
    similarity = result["similarity"]
    result_id = result["result_id"]

    st.markdown("---")
    st.header("📊 Your Skill Analysis")

    if saved_name:
        st.write(f"Welcome back, **{saved_name}**!")

    st.write(f"**Analyzed target role:** {role}")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Skill coverage",
        f"{coverage}%",
    )

    metric2.metric(
        "Skills matched",
        f"{len(matched)} / {len(required_skills)}",
    )

    metric3.metric(
        "Skills to learn",
        len(missing),
    )

    st.progress(min(int(coverage), 100))

    st.caption(
        "Coverage = matched listed skills ÷ total listed required skills. "
        "It is not a probability of getting hired."
    )

    # -----------------------------------------------------
    # MATCHED AND MISSING SKILLS
    # -----------------------------------------------------

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("🟢 Skills You Have")

        if matched:
            for skill in matched:
                st.success(f"✓ {skill}")
        else:
            st.write("No exact matches yet.")

    with right:
        st.subheader("🟡 Skills to Learn")

        if missing:
            for skill in missing:
                st.warning(f"＋ {skill}")
        else:
            st.success("You've matched all listed required skills!")

    # -----------------------------------------------------
    # PERSISTENT CHECKLIST
    # Checkbox state is maintained across Streamlit reruns.
    # Each analysis has a stable, separate set of checkbox keys.
    # -----------------------------------------------------

    st.markdown("---")
    st.header("📚 Your Learning Roadmap")

    st.write(
        "Tick a skill when you have learned and practiced it. "
        "Your progress updates automatically."
    )

    if missing:
        completed_count = 0

        for skill in missing:
            checkbox_key = (
                f"learn_{result_id}_{normalize_skill(skill)}"
            )

            # Streamlit stores the checkbox value in session_state.
            # Do not supply value= when using a stateful key.
            is_completed = st.checkbox(
                f"Learn and practice {skill}",
                key=checkbox_key,
            )

            if is_completed:
                completed_count += 1

        total_missing = len(missing)

        progress = (
            completed_count / total_missing
            if total_missing else 1.0
        )

        st.progress(progress)

        st.markdown(
            f"**Learning progress:** {completed_count} of "
            f"{total_missing} skills completed "
            f"({round(progress * 100)}%)"
        )

        if completed_count == total_missing:
            st.success(
                "🎉 Great work! You have checked off every skill "
                "in this learning checklist."
            )

    else:
        st.success(
            "No listed skill gaps. Consider building a project "
            "to demonstrate your abilities."
        )

    # -----------------------------------------------------
    # TEXT SIMILARITY
    # -----------------------------------------------------

    st.markdown("---")
    st.header("🤖 Text Similarity Analysis")

    st.metric(
        "TF-IDF Cosine Similarity",
        f"{similarity}%",
    )

    st.caption(
        "TF-IDF cosine similarity measures text similarity between "
        "the skills you entered and the required skills. It is "
        "different from exact skill coverage and does not predict "
        "your hiring chances."
    )

    # -----------------------------------------------------
    # RECOMMEND OTHER ROLES
    # -----------------------------------------------------

    st.markdown("---")
    st.header("🧭 Explore Other Job Roles")

    recommendations = []

    for _, row in jobs.iterrows():
        other_role = row["job_role"]

        if other_role == role:
            continue

        other_required = parse_skills(row["required_skills"])

        other_matched, other_missing = get_skill_match(
            user_skills,
            other_required,
        )

        other_coverage = (
            round(len(other_matched) / len(other_required) * 100, 1)
            if other_required else 0
        )

        recommendations.append({
            "Job Role": other_role,
            "Skill Coverage (%)": other_coverage,
            "Matched Skills": len(other_matched),
            "Skills to Learn": len(other_missing),
        })

    recommendations.sort(
        key=lambda item: item["Skill Coverage (%)"],
        reverse=True,
    )

    if recommendations:
        recommendation_df = pd.DataFrame(recommendations)

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skill Coverage (%)": st.column_config.ProgressColumn(
                    "Skill Coverage",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "Matched Skills": st.column_config.NumberColumn(
                    "Matched Skills",
                    format="%d",
                ),
                "Skills to Learn": st.column_config.NumberColumn(
                    "Skills to Learn",
                    format="%d",
                ),
            },
        )
    else:
        st.write("Add more job roles to jobs.csv to compare careers.")

    # -----------------------------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------------------------

    st.markdown("---")
    st.header("📄 Download Your Report")

    report = f"""SKILLCHECK - SKILL GAP REPORT
================================

Name: {saved_name or 'Not provided'}
Target Role: {role}

Your Skills:
{', '.join(user_skills)}

Matched Skills:
{', '.join(matched) or 'None'}

Skills to Learn:
{', '.join(missing) or 'None'}

Required Skill Coverage: {coverage}%
TF-IDF Cosine Similarity: {similarity}%

Learning Checklist:
"""

    for skill in missing:
        checkbox_key = f"learn_{result_id}_{normalize_skill(skill)}"
        completed = st.session_state.get(checkbox_key, False)
        status = "Completed" if completed else "Not completed"
        report += f"\n- {skill}: {status}"

    report += """

Note:
Skill coverage and TF-IDF similarity are estimates based on the
sample job dataset. They do not predict hiring outcomes.
"""

    st.download_button(
        "⬇️ Download My Skill Report",
        data=report,
        file_name="skillcheck_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        "Your checklist is saved for this browser session. "
        "A server restart or a new session may clear session progress."
    )

else:
    st.markdown("---")
    st.subheader("✨ Your career roadmap starts here")
    st.write(
        "Enter your skills and click **Analyze My Skills** to see "
        "your matches, skill gaps, career comparisons, and learning checklist."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "SkillCheck | Career Intelligence & Skill Gap Analyzer | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
