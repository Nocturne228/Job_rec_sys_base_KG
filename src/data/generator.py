"""
Generate mock data for the job recommendation system.
"""
import random
from typing import List, Dict
from datetime import datetime, timedelta

from .models import (
    Skill, User, JobPosting, Application, Interaction, SkillLevel, GraphEntities
)


# Predefined skills
SKILLS_DATA = [
    Skill(id="python", name="Python", category="programming", description="Python programming language"),
    Skill(id="java", name="Java", category="programming", description="Java programming language"),
    Skill(id="javascript", name="JavaScript", category="programming", description="JavaScript programming language"),
    Skill(id="react", name="React", category="frontend", description="React library for building user interfaces"),
    Skill(id="vue", name="Vue.js", category="frontend", description="Vue.js framework"),
    Skill(id="nodejs", name="Node.js", category="backend", description="Node.js runtime"),
    Skill(id="docker", name="Docker", category="devops", description="Containerization platform"),
    Skill(id="kubernetes", name="Kubernetes", category="devops", description="Container orchestration"),
    Skill(id="aws", name="AWS", category="cloud", description="Amazon Web Services"),
    Skill(id="azure", name="Azure", category="cloud", description="Microsoft Azure"),
    Skill(id="gcp", name="Google Cloud", category="cloud", description="Google Cloud Platform"),
    Skill(id="sql", name="SQL", category="database", description="Structured Query Language"),
    Skill(id="mongodb", name="MongoDB", category="database", description="NoSQL database"),
    Skill(id="postgresql", name="PostgreSQL", category="database", description="Relational database"),
    Skill(id="pytorch", name="PyTorch", category="ml", description="Machine learning framework"),
    Skill(id="tensorflow", name="TensorFlow", category="ml", description="Machine learning framework"),
    Skill(id="scikit", name="Scikit-learn", category="ml", description="Machine learning library"),
    Skill(id="pandas", name="Pandas", category="data", description="Data manipulation library"),
    Skill(id="numpy", name="NumPy", category="data", description="Numerical computing library"),
    Skill(id="communication", name="Communication", category="soft", description="Verbal and written communication"),
    Skill(id="leadership", name="Leadership", category="soft", description="Team leadership and management"),
    Skill(id="problem_solving", name="Problem Solving", category="soft", description="Analytical problem solving"),
]

SKILL_LEVELS = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED, SkillLevel.EXPERT]

# Sample user names
USER_NAMES = [
    "Alice Chen", "Bob Smith", "Charlie Brown", "Diana Wang", "Ethan Johnson",
    "Fiona Garcia", "George Miller", "Hannah Davis", "Ian Wilson", "Julia Martinez",
    "Kevin Lee", "Lisa Taylor", "Michael Brown", "Nancy Clark", "Oliver Rodriguez",
    "Patricia Lewis", "Quinn Walker", "Rachel Hall", "Samuel Allen", "Tina Young"
]

# Sample companies
COMPANIES = [
    "TechCorp", "DataSys", "CloudNet", "SoftWorks", "AI Innovations",
    "WebSolutions", "MobileFirst", "CyberSecure", "BigData Inc", "DevOps Pro"
]

# Job titles
JOB_TITLES = [
    "Software Engineer", "Data Scientist", "DevOps Engineer", "Frontend Developer",
    "Backend Developer", "Full Stack Developer", "Machine Learning Engineer",
    "Cloud Architect", "Database Administrator", "Product Manager"
]

# Education levels
EDUCATION_LEVELS = [
    "Bachelor's in Computer Science", "Master's in Data Science", "PhD in Machine Learning",
    "Bachelor's in Engineering", "Bootcamp Graduate", "Self-taught"
]


def generate_skills() -> List[Skill]:
    """Generate predefined skills."""
    return SKILLS_DATA


def generate_users(num_users: int = 20) -> List[User]:
    """Generate mock users."""
    users = []
    skill_ids = [skill.id for skill in SKILLS_DATA]

    for i in range(num_users):
        user_id = f"user_{i+1:03d}"
        name = USER_NAMES[i % len(USER_NAMES)]
        education = random.choice(EDUCATION_LEVELS)
        experience_years = random.uniform(0.5, 10.0)

        # Generate user skills (3-8 skills per user)
        num_skills = random.randint(3, 8)
        user_skill_ids = random.sample(skill_ids, num_skills)
        skills = {skill_id: random.choice(SKILL_LEVELS) for skill_id in user_skill_ids}

        # Generate resume text
        skill_names = [skill.name for skill in SKILLS_DATA if skill.id in user_skill_ids]
        resume_text = f"{name} has {experience_years:.1f} years of experience in {', '.join(skill_names)}."

        users.append(User(
            id=user_id,
            name=name,
            education=education,
            experience_years=experience_years,
            skills=skills,
            resume_text=resume_text
        ))

    return users


def generate_jobs(num_jobs: int = 50) -> List[JobPosting]:
    """Generate mock job postings."""
    jobs = []
    skill_ids = [skill.id for skill in SKILLS_DATA]

    for i in range(num_jobs):
        job_id = f"job_{i+1:03d}"
        title = random.choice(JOB_TITLES)
        company = random.choice(COMPANIES)

        # Generate job description
        description = f"We are looking for a {title} to join our team at {company}. "
        description += "The ideal candidate will have strong technical skills and be a team player."

        # Generate required skills (2-5 skills)
        num_required = random.randint(2, 5)
        required_skill_ids = random.sample(skill_ids, num_required)
        required_skills = {skill_id: random.choice(SKILL_LEVELS) for skill_id in required_skill_ids}

        # Generate preferred skills (0-3 skills)
        num_preferred = random.randint(0, 3)
        preferred_skill_ids = random.sample([s for s in skill_ids if s not in required_skill_ids],
                                           min(num_preferred, len(skill_ids) - num_required))
        preferred_skills = {skill_id: random.choice(SKILL_LEVELS) for skill_id in preferred_skill_ids}

        # Generate salary range (in thousands)
        min_salary = random.uniform(60, 120)
        max_salary = min_salary + random.uniform(20, 60)

        jobs.append(JobPosting(
            id=job_id,
            title=title,
            company=company,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            salary_range=(min_salary, max_salary)
        ))

    return jobs


def generate_applications(users: List[User], jobs: List[JobPosting],
                         application_rate: float = 0.3) -> List[Application]:
    """Generate mock applications."""
    applications = []
    statuses = ["applied", "interview", "rejected", "accepted"]

    for user in users:
        # Each user applies to some jobs
        for job in jobs:
            if random.random() < application_rate:
                status = random.choices(
                    statuses,
                    weights=[0.5, 0.2, 0.2, 0.1]  # Higher weight for "applied"
                )[0]

                # Generate a date within the last 6 months
                days_ago = random.randint(0, 180)
                date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

                applications.append(Application(
                    user_id=user.id,
                    job_id=job.id,
                    status=status,
                    date=date
                ))

    return applications


def generate_interactions(users: List[User], jobs: List[JobPosting],
                         interaction_rate: float = 0.5) -> List[Interaction]:
    """Generate mock interactions (views, clicks, saves)."""
    interactions = []
    interaction_types = ["view", "click", "save", "apply"]
    weights = [0.4, 0.3, 0.2, 0.1]  # More views than applies

    for user in users:
        for job in jobs:
            if random.random() < interaction_rate:
                interaction_type = random.choices(interaction_types, weights=weights)[0]

                # Generate timestamp within the last 30 days
                days_ago = random.randint(0, 30)
                hours_ago = random.randint(0, 23)
                minutes_ago = random.randint(0, 59)
                timestamp = (datetime.now() - timedelta(days=days_ago,
                                                       hours=hours_ago,
                                                       minutes=minutes_ago)).isoformat()

                interactions.append(Interaction(
                    user_id=user.id,
                    job_id=job.id,
                    interaction_type=interaction_type,
                    timestamp=timestamp
                ))

    return interactions


def generate_mock_data(num_users: int = 20, num_jobs: int = 50) -> GraphEntities:
    """Generate complete mock dataset."""
    skills = generate_skills()
    users = generate_users(num_users)
    jobs = generate_jobs(num_jobs)
    applications = generate_applications(users, jobs, application_rate=0.2)
    interactions = generate_interactions(users, jobs, interaction_rate=0.3)

    return GraphEntities(
        users=users,
        jobs=jobs,
        skills=skills,
        applications=applications,
        interactions=interactions
    )