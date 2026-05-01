"""
Personal information encryption module.
AES-256-CBC encryption for privacy-sensitive fields (name, phone, email, address).
"""
import os
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def _derive_key(password: str, salt: bytes = b"jobrec_salt") -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=32)


def encrypt_personal_info(plaintext: str, master_password: str) -> str:
    key = _derive_key(master_password)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded = plaintext.encode() + b"\x00" * (16 - len(plaintext.encode()) % 16)
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext).decode()


def decrypt_personal_info(encrypted_b64: str, master_password: str) -> str:
    key = _derive_key(master_password)
    raw = base64.b64decode(encrypted_b64)
    iv, ciphertext = raw[:16], raw[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    return padded.rstrip(b"\x00").decode()


SENSITIVE_FIELDS = ["name", "phone", "email", "address"]

def encrypt_user_profile(user_dict: dict, master_password: str) -> dict:
    encrypted = dict(user_dict)
    for field in SENSITIVE_FIELDS:
        if field in encrypted and encrypted[field]:
            encrypted[field] = encrypt_personal_info(str(encrypted[field]), master_password)
    encrypted["_encrypted_fields"] = [f for f in SENSITIVE_FIELDS if f in user_dict]
    return encrypted


def decrypt_user_profile(encrypted_dict: dict, master_password: str) -> dict:
    decrypted = dict(encrypted_dict)
    for field in encrypted_dict.get("_encrypted_fields", []):
        if field in decrypted and decrypted[field]:
            decrypted[field] = decrypt_personal_info(decrypted[field], master_password)
    return decrypted
