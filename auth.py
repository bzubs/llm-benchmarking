import sqlite3
import bcrypt
from dotenv import load_dotenv
import os

load_dotenv()

DB_NAME = "users.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def register_user(username, password, access_code):

    access_code_env = os.getenv("ACCESS_CODE")
    if access_code == access_code_env:
        conn = get_connection()
        cursor = conn.cursor()

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed)
            )
            conn.commit()
            return True, "User registered"
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        finally:
            conn.close()
    else:
        return False, "Incorrect Access Code"


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT password FROM users WHERE username = ?",
        (username,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_hash = result[0]
        if bcrypt.checkpw(password.encode(), stored_hash):
            return True, "Login successful"

    return False, "Invalid credentials"

