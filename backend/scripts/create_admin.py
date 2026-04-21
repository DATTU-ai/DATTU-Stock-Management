"""
Create Initial Admin User
=========================
Run this script ONCE to create the first admin account.

Usage:
    python scripts/create_admin.py

This script will prompt for admin credentials and create the account.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from getpass import getpass

import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dattu_bill")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


async def create_admin():
    """Create the initial admin user."""
    print("\n" + "="*50)
    print("  DATTU BILL - Create Admin Account")
    print("="*50 + "\n")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]
    users_collection = db["users"]
    
    # Check if any admin already exists
    existing_admin = await users_collection.find_one({"role": "admin"})
    if existing_admin:
        print(f"⚠️  An admin account already exists: {existing_admin['username']}")
        response = input("Do you want to create another admin? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Get admin details
    print("Enter details for the new admin account:\n")
    
    username = input("Username: ").strip()
    if not username or len(username) < 3:
        print("❌ Username must be at least 3 characters.")
        return
    
    # Check if username exists
    existing_user = await users_collection.find_one({"username": username})
    if existing_user:
        print(f"❌ Username '{username}' already exists.")
        return
    
    email = input("Email: ").strip()
    if not email or "@" not in email:
        print("❌ Invalid email address.")
        return
    
    # Check if email exists
    existing_email = await users_collection.find_one({"email": email})
    if existing_email:
        print(f"❌ Email '{email}' already exists.")
        return
    
    password = getpass("Password (min 6 chars): ")
    if len(password) < 6:
        print("❌ Password must be at least 6 characters.")
        return
    
    confirm_password = getpass("Confirm Password: ")
    if password != confirm_password:
        print("❌ Passwords do not match.")
        return
    
    # Create admin user
    admin_doc = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": "system"
    }
    
    result = await users_collection.insert_one(admin_doc)
    
    print("\n" + "="*50)
    print("  ✅ Admin account created successfully!")
    print("="*50)
    print(f"  Username: {username}")
    print(f"  Email: {email}")
    print(f"  Role: admin")
    print(f"  ID: {result.inserted_id}")
    print("="*50 + "\n")
    
    # Close connection
    client.close()


if __name__ == "__main__":
    asyncio.run(create_admin())
