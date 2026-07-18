"""Authenticated encryption helpers for privacy-sensitive profile fields."""

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MAGIC = b"JRG1"
_SALT_BYTES = 16
_NONCE_BYTES = 12
_KDF_ITERATIONS = 310_000


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("master_password must not be empty")
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _KDF_ITERATIONS, dklen=32
    )


def encrypt_personal_info(plaintext: str, master_password: str) -> str:
    salt = os.urandom(_SALT_BYTES)
    nonce = os.urandom(_NONCE_BYTES)
    key = _derive_key(master_password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), _MAGIC)
    return base64.urlsafe_b64encode(_MAGIC + salt + nonce + ciphertext).decode("ascii")


def decrypt_personal_info(encrypted_b64: str, master_password: str) -> str:
    raw = base64.urlsafe_b64decode(encrypted_b64.encode("ascii"))
    minimum = len(_MAGIC) + _SALT_BYTES + _NONCE_BYTES + 16
    if len(raw) < minimum or raw[: len(_MAGIC)] != _MAGIC:
        raise ValueError("Unsupported or malformed encrypted payload")
    offset = len(_MAGIC)
    salt = raw[offset : offset + _SALT_BYTES]
    offset += _SALT_BYTES
    nonce = raw[offset : offset + _NONCE_BYTES]
    ciphertext = raw[offset + _NONCE_BYTES :]
    key = _derive_key(master_password, salt)
    return AESGCM(key).decrypt(nonce, ciphertext, _MAGIC).decode("utf-8")


SENSITIVE_FIELDS = ["name", "phone", "email", "address"]


def encrypt_user_profile(user_dict: dict, master_password: str) -> dict:
    encrypted = dict(user_dict)
    for field in SENSITIVE_FIELDS:
        if field in encrypted and encrypted[field]:
            encrypted[field] = encrypt_personal_info(
                str(encrypted[field]), master_password
            )
    encrypted["_encrypted_fields"] = [f for f in SENSITIVE_FIELDS if f in user_dict]
    return encrypted


def decrypt_user_profile(encrypted_dict: dict, master_password: str) -> dict:
    decrypted = dict(encrypted_dict)
    for field in encrypted_dict.get("_encrypted_fields", []):
        if field in decrypted and decrypted[field]:
            decrypted[field] = decrypt_personal_info(decrypted[field], master_password)
    return decrypted
