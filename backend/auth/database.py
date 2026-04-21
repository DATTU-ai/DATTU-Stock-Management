"""
Database Connection Module
==========================
Handles async MongoDB connection using Motor driver.
"""

import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_BACKEND_ENV_PATH)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dattu_bill")

# Global database client (lazy initialization)
_client: AsyncIOMotorClient = None
_database = None


async def get_database():
    """
    Get the MongoDB database instance.
    Creates connection on first call (lazy initialization).
    
    Returns:
        AsyncIOMotorDatabase: The database instance
    """
    global _client, _database
    
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URL)
        _database = _client[MONGODB_DB_NAME]
        print(f"[DATABASE] Connected to MongoDB: {MONGODB_DB_NAME}")
    
    return _database


async def get_users_collection():
    """
    Get the users collection.
    
    Returns:
        AsyncIOMotorCollection: The users collection
    """
    db = await get_database()
    return db["users"]


async def close_database():
    """Close the database connection."""
    global _client, _database
    
    if _client is not None:
        _client.close()
        _client = None
        _database = None
        print("[DATABASE] MongoDB connection closed")
