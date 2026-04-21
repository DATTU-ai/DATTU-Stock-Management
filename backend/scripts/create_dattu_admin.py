"""
Create or update the DATTU admin account.

This script provisions a single admin user with:
- username: DATTU
- password: admin123

The email is set to a safe placeholder value unless the user already exists.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


# Add backend directory to path so we can import local helpers if needed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dattu_bill")

USERNAME = "DATTU"
EMAIL = "dattu@example.com"
PASSWORD = "admin123"


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


async def create_or_update_admin():
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]
    users_collection = db["users"]

    existing_user = await users_collection.find_one({"username": USERNAME})
    payload = {
        "username": USERNAME,
        "email": EMAIL if not existing_user else existing_user.get("email", EMAIL),
        "password_hash": hash_password(PASSWORD),
        "role": "admin",
        "is_active": True,
        "is_logged_in": False,
        "last_activity": None,
        "created_at": existing_user.get("created_at") if existing_user else datetime.utcnow(),
        "created_by": existing_user.get("created_by", "system") if existing_user else "system",
    }

    if existing_user:
        result = await users_collection.update_one(
            {"username": USERNAME},
            {"$set": payload}
        )
        action = "updated"
    else:
        result = await users_collection.insert_one(payload)
        action = "created"

    print("\n" + "=" * 60)
    print(f"Admin account {action} successfully")
    print("=" * 60)
    print(f"Username: {USERNAME}")
    print(f"Password: {PASSWORD}")
    print(f"Role: admin")
    print(f"Email: {payload['email']}")
    print(f"Database: {MONGODB_DB_NAME}")
    print(f"Result ID: {getattr(result, 'inserted_id', USERNAME)}")
    print("=" * 60 + "\n")

    client.close()


if __name__ == "__main__":
    asyncio.run(create_or_update_admin())
