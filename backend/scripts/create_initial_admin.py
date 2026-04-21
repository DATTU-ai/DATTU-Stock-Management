import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor.motor_asyncio import AsyncIOMotorClient
from auth.security import get_password_hash
from datetime import datetime

# Prompt for details
print("="*60)
print("CREATE INITIAL ADMIN USER")
print("="*60)
print("This script will create a new admin user in your MongoDB database.")
print("You can use the connection string from your local .env or paste one.")
print("-" * 60)

mongo_url = input("Enter MongoDB URL (press Enter to use from .env): ").strip()
if not mongo_url:
    from dotenv import load_dotenv
    _BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_BACKEND_ENV_PATH)
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

print(f"Using MongoDB URL: {mongo_url}")

username = input("Enter new Admin Username: ").strip()
email = input("Enter new Admin Email: ").strip()
password = input("Enter new Admin Password: ").strip()

if not all([username, email, password]):
    print("Error: All fields are required!")
    sys.exit(1)

async def create_user():
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client.get_default_database("dattu_bill")
        users_collection = db["users"]
        
        # Check existing
        if await users_collection.find_one({"username": username}):
            print(f"Error: User '{username}' already exists!")
            return
            
        user_doc = {
            "username": username,
            "email": email,
            "password_hash": get_password_hash(password),
            "role": "admin",
            "is_active": True,
            "is_logged_in": False,
            "last_activity": None,
            "created_at": datetime.utcnow(),
            "created_by": "system_init"
        }
        
        await users_collection.insert_one(user_doc)
        print(f"\nSUCCESS! User '{username}' created.")
        print("You can now login with these credentials.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(create_user())
