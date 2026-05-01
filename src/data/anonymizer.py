"""
Data anonymization: replaces personal identifiers with pseudonyms before storage.
References: GDPR data-minimization principle, competition QR-3 requirement.
"""
import hashlib
import re


def anonymize_email(email: str) -> str:
    user, domain = email.split("@", 1)
    return f"{user[:2]}***@{domain}"


def anonymize_phone(phone: str) -> str:
    return phone[:3] + "****" + phone[-4:]


def anonymize_name(name: str) -> str:
    parts = name.split()
    if len(parts) == 1:
        return parts[0][0] + "***"
    return parts[0][0] + "*** " + parts[-1][0] + "***"


def pseudonymize(user_id: str, salt: str = "jobrec") -> str:
    return hashlib.sha256(f"{user_id}:{salt}".encode()).hexdigest()[:12]


def anonymize_resume(resume_text: str) -> str:
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', resume_text)
    text = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{17}[\dXx]\b', '[ID_NUMBER]', text)
    return text
