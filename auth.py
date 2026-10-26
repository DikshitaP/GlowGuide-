"""Tiny username/password auth for the Streamlit app (SQLite + salted PBKDF2 hashes, standard library only)."""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time

DB_PATH = os.environ.get("AUTH_DB", "users.db")
ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY COLLATE NOCASE, "
              "salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created REAL NOT NULL)")
    return c


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def create_user(username: str, password: str):
    """Returns an error message, or None on success."""
    username = username.strip()
    if not USERNAME_RE.match(username):
        return "Username must be 3 to 30 characters: letters, numbers, dot, dash or underscore."
    if len(password) < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    salt = secrets.token_bytes(16)
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (username, salt, _hash(password, salt), time.time()))
    except sqlite3.IntegrityError:
        return "That username is already taken."
    return None


def verify_user(username: str, password: str):
    """Returns the stored username (with its original capitalisation) if the credentials are right, else None."""
    with _conn() as c:
        row = c.execute("SELECT username, salt, pw_hash FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        _hash(password, b"\0" * 16)          # same work as a real check, so timing doesn't reveal which usernames exist
        return None
    name, salt, stored = row
    return name if hmac.compare_digest(_hash(password, salt), stored) else None
