# SkillCheck

SkillCheck is a Streamlit career-intelligence app that compares a user's
current skills with the requirements for selected job roles.

It provides:

- Skill coverage and matched-skill metrics
- A list of skills to learn
- A learning roadmap with progress tracking
- TF-IDF text similarity analysis
- Exploration of other job roles
- A downloadable analysis report

## Requirements

- Python 3.10 or newer
- The job data file `jobs.csv`

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

The app expects `jobs.csv` in the project root with these columns:

```text
job_role,required_skills
```

`required_skills` should contain comma-separated skills for each job role.

## Project files

- `app.py` - Streamlit application
- `style.css` - Custom application styling
- `jobs.csv` - Job roles and required skills
- `requirements.txt` - Python dependencies
