"""
View Database Contents
======================
Quick script to see what's in the MongoDB database.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)


async def view_database():
    print("\n" + "="*50)
    print("  DATTU_BILL - Database Viewer")
    print("="*50)
    
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "dattu_bill")

    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    print(f"\nUsing database: {db_name}")
    
    # Show all collections
    collections = await db.list_collection_names()
    print(f"\nCollections: {collections}")
    
    # Show users
    users_collection = db["users"]
    user_count = await users_collection.count_documents({})
    print(f"\nTotal Users: {user_count}")
    print("-"*50)
    
    async for user in users_collection.find():
        print(f"  Username: {user['username']}")
        print(f"  Email: {user['email']}")
        print(f"  Role: {user['role']}")
        print(f"  Active: {user.get('is_active', True)}")
        print(f"  Created: {user.get('created_at', 'N/A')}")
        print("-"*50)
    
    client.close()
    print()

if __name__ == "__main__":
    asyncio.run(view_database())
