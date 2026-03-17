"""
routers/users.py — user management

Routes:
  POST /users/register
  POST /users/login
  GET  /users/me
  PUT  /users/me
  GET  /users/me/settings
  PUT  /users/me/settings
  GET  /users/me/stats
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from core.state  import app_state
from core.auth   import get_current_user, hash_password, verify_password, create_access_token
from schemas.models import (
    RegisterRequest, LoginRequest, AuthResponse,
    UserResponse, UpdateProfileRequest,
    UserSettingsResponse, UpdateSettingsRequest,
    UserStatsResponse, MessageResponse,
)

router = APIRouter()
log    = logging.getLogger("s2s.users")


# ── register ─────────────────────────────────────────────────────────
@router.post("/register", response_model=AuthResponse, status_code=201,
             summary="Create a new account")
async def register(body: RegisterRequest):
    existing = await app_state.db_pool.fetchrow(
        "SELECT id FROM users WHERE email=$1", body.email
    )
    if existing:
        raise HTTPException(409, "Email already registered")

    user_id      = uuid.uuid4()
    hashed_pw    = hash_password(body.password)

    async with app_state.db_pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("""
                INSERT INTO users (id, name, email, password_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING id, name, email, plan, created_at
            """, user_id, body.name, body.email, hashed_pw)

            await conn.execute("""
                INSERT INTO user_settings (user_id) VALUES ($1)
                ON CONFLICT DO NOTHING
            """, user_id)

            await conn.execute("""
                INSERT INTO user_stats (user_id) VALUES ($1)
                ON CONFLICT DO NOTHING
            """, user_id)

    token = create_access_token(str(row["id"]), row["email"])
    user  = UserResponse(**dict(row))
    return AuthResponse(access_token=token, user=user)


# ── login ─────────────────────────────────────────────────────────────
@router.post("/login", response_model=AuthResponse, summary="Sign in")
async def login(body: LoginRequest):
    row = await app_state.db_pool.fetchrow(
        "SELECT id, name, email, plan, password_hash, created_at FROM users WHERE email=$1",
        body.email,
    )
    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(str(row["id"]), row["email"])
    user  = UserResponse(
        id=row["id"], name=row["name"], email=row["email"],
        plan=row["plan"], created_at=row["created_at"],
    )
    return AuthResponse(access_token=token, user=user)


# ── me ────────────────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse, summary="Get current user profile")
async def get_me(current_user: dict = Depends(get_current_user)):
    row = await app_state.db_pool.fetchrow(
        "SELECT id, name, email, plan, created_at FROM users WHERE id=$1",
        current_user["id"],
    )
    return UserResponse(**dict(row))


@router.put("/me", response_model=UserResponse, summary="Update profile")
async def update_me(
    body:         UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
):
    updates, params, i = [], [], 1

    if body.name is not None:
        updates.append(f"name=${i}"); params.append(body.name); i+=1
    if body.avatar_url is not None:
        updates.append(f"avatar_url=${i}"); params.append(body.avatar_url); i+=1

    if not updates:
        raise HTTPException(400, "No fields to update")

    params.append(current_user["id"])
    row = await app_state.db_pool.fetchrow(
        f"UPDATE users SET {', '.join(updates)}, updated_at=NOW() "
        f"WHERE id=${i} RETURNING id, name, email, plan, created_at",
        *params,
    )
    return UserResponse(**dict(row))


# ── settings ──────────────────────────────────────────────────────────
@router.get("/me/settings", response_model=UserSettingsResponse,
            summary="Get user settings")
async def get_settings(current_user: dict = Depends(get_current_user)):
    row = await app_state.db_pool.fetchrow(
        "SELECT * FROM user_settings WHERE user_id=$1", current_user["id"]
    )
    if not row:
        raise HTTPException(404, "Settings not found")
    return UserSettingsResponse(**dict(row))


@router.put("/me/settings", response_model=UserSettingsResponse,
            summary="Update user settings")
async def update_settings(
    body:         UpdateSettingsRequest,
    current_user: dict = Depends(get_current_user),
):
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(400, "No settings to update")

    updates = [f"{k}=${i+1}" for i, k in enumerate(fields)]
    params  = list(fields.values()) + [current_user["id"]]

    row = await app_state.db_pool.fetchrow(
        f"UPDATE user_settings SET {', '.join(updates)}, updated_at=NOW() "
        f"WHERE user_id=${len(params)} RETURNING *",
        *params,
    )
    return UserSettingsResponse(**dict(row))


# ── stats ─────────────────────────────────────────────────────────────
@router.get("/me/stats", response_model=UserStatsResponse,
            summary="Get usage statistics")
async def get_stats(current_user: dict = Depends(get_current_user)):
    row = await app_state.db_pool.fetchrow(
        "SELECT * FROM user_stats WHERE user_id=$1", current_user["id"]
    )
    if not row:
        return UserStatsResponse(
            total_translations=0, total_words_translated=0,
            total_audio_seconds=0, avg_asr_confidence=None,
            most_used_target_lang=None, streak_days=0, last_active_at=None,
        )
    return UserStatsResponse(**dict(row))