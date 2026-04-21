"""
Authentication Routes
=====================
Handles user login, registration (admin only), and user management.
Implements single-session enforcement (one device at a time).
"""

from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from auth.database import get_users_collection
from auth.security import verify_password, get_password_hash, create_access_token
from auth.dependencies import get_current_user, get_current_admin_user
from models.user import UserCreate, UserResponse, Token

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Session timeout in hours (user can login again after this time of inactivity)
SESSION_TIMEOUT_HOURS = 24


def is_session_active(user: dict) -> bool:
    """
    Check if user has an active session that hasn't expired.
    
    Returns True if user is currently logged in and session is still valid.
    """
    if not user.get("is_logged_in", False):
        return False
    
    last_activity = user.get("last_activity")
    if not last_activity:
        return False
    
    # Check if session has expired (24 hours of inactivity)
    expiry_time = last_activity + timedelta(hours=SESSION_TIMEOUT_HOURS)
    if datetime.utcnow() > expiry_time:
        return False  # Session expired
    
    return True  # Session is still active


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(data: LoginRequest):
    """
    Authenticate user and return JWT token.
    
    Enforces single-session: user cannot login if already logged in elsewhere.
    Sessions expire after 24 hours of inactivity.
    """
    users_collection = await get_users_collection()
    
    # Find user by username
    user = await users_collection.find_one({"username": data.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Check if user is already logged in on another device
    # EXCEPTION: Admins can login on multiple devices
    if user.get("role") != "admin" and is_session_active(user):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Already logged in on another device"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user["username"]})
    
    # Update user's session status in database
    await users_collection.update_one(
        {"username": user["username"]},
        {
            "$set": {
                "is_logged_in": True,
                "last_activity": datetime.utcnow(),
                "login_time": datetime.utcnow()
            }
        }
    )
    
    # Return token with user info (excluding sensitive data)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout the current user and clear their session.
    """
    users_collection = await get_users_collection()
    
    # Clear session status
    await users_collection.update_one(
        {"username": current_user["username"]},
        {
            "$set": {
                "is_logged_in": False,
                "last_activity": None,
                "logout_time": datetime.utcnow()
            }
        }
    )
    
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user's information.
    Also updates last_activity to keep session alive.
    """
    # Update last activity (keeps session alive)
    users_collection = await get_users_collection()
    await users_collection.update_one(
        {"username": current_user["username"]},
        {"$set": {"last_activity": datetime.utcnow()}}
    )
    
    return {
        "id": str(current_user["_id"]),
        "username": current_user["username"],
        "email": current_user["email"],
        "role": current_user["role"],
        "is_active": current_user.get("is_active", True),
        "created_at": current_user.get("created_at")
    }


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Register a new user (Admin only).
    
    Only administrators can create new user accounts.
    """
    users_collection = await get_users_collection()
    
    # Check if username already exists
    existing_user = await users_collection.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await users_collection.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user document
    user_doc = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": get_password_hash(user_data.password),
        "role": user_data.role,
        "is_active": True,
        "is_logged_in": False,
        "last_activity": None,
        "created_at": datetime.utcnow(),
        "created_by": current_admin["username"]
    }
    
    # Insert into database
    result = await users_collection.insert_one(user_doc)
    
    return {
        "message": "User created successfully",
        "user": {
            "id": str(result.inserted_id),
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role
        }
    }


@router.get("/users", response_model=List[dict])
async def list_users(current_admin: dict = Depends(get_current_admin_user)):
    """
    List all users (Admin only).
    """
    users_collection = await get_users_collection()
    
    users = []
    async for user in users_collection.find():
        users.append({
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "is_active": user.get("is_active", True),
            "is_logged_in": is_session_active(user),
            "last_activity": user.get("last_activity"),
            "created_at": user.get("created_at")
        })
    
    return users


@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Delete a user (Admin only).
    
    Admins cannot delete themselves.
    """
    if username == current_admin["username"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    users_collection = await get_users_collection()
    
    result = await users_collection.delete_one({"username": username})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": f"User '{username}' deleted successfully"}


@router.patch("/users/{username}/toggle-active")
async def toggle_user_active(
    username: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Enable/Disable a user account (Admin only).
    
    Disabling a user also logs them out.
    """
    if username == current_admin["username"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account"
        )
    
    users_collection = await get_users_collection()
    
    user = await users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    new_status = not user.get("is_active", True)
    
    # If disabling, also log them out
    update_data = {"is_active": new_status}
    if not new_status:
        update_data["is_logged_in"] = False
        update_data["last_activity"] = None
    
    await users_collection.update_one(
        {"username": username},
        {"$set": update_data}
    )
    
    return {
        "message": f"User '{username}' {'enabled' if new_status else 'disabled'} successfully",
        "is_active": new_status
    }

