"""
routers/history.py — translation history

Routes:
  GET    /history               List history feed
  GET    /history/{id}          Get single item
  PATCH  /history/{id}          Star / tag an item
  DELETE /history/{id}          Soft delete
  DELETE /history               Clear all history
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from core.state  import app_state
from core.auth   import get_current_user
from schemas.models import (
    HistoryListResponse, HistoryItem,
    UpdateHistoryRequest, MessageResponse,
)

router = APIRouter()
log    = logging.getLogger("s2s.history")


@router.get("", response_model=HistoryListResponse, summary="List translation history")
async def list_history(
    limit:       int           = Query(20, ge=1, le=100),
    offset:      int           = Query(0,  ge=0),
    starred_only: bool         = Query(False),
    target_lang: Optional[str] = Query(None, description="Filter by target language: fr or tw"),
    current_user: dict         = Depends(get_current_user),
):
    uid = current_user["id"]

    where_parts = ["h.user_id = $1", "h.is_deleted = FALSE"]
    params: list = [uid]
    i = 2

    if starred_only:
        where_parts.append("h.is_starred = TRUE")

    if target_lang:
        where_parts.append(f"s.target_language_code = ${i}")
        params.append(target_lang); i += 1

    where = " AND ".join(where_parts)

    # count
    total = await app_state.db_pool.fetchval(
        f"""SELECT COUNT(*) FROM translation_history h
            JOIN translation_sessions s ON s.id = h.session_id
            WHERE {where}""",
        *params,
    )

    # fetch page
    params += [limit, offset]
    rows = await app_state.db_pool.fetch(f"""
        SELECT * FROM v_history_feed
        WHERE user_id = $1
        {"AND is_starred = TRUE" if starred_only else ""}
        {"AND target_language = (SELECT name FROM languages WHERE code = $2)" if target_lang else ""}
        LIMIT ${len(params)-1} OFFSET ${len(params)}
    """, *params)

    items = [HistoryItem(**dict(r)) for r in rows]
    return HistoryListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{history_id}", response_model=HistoryItem, summary="Get history item")
async def get_history_item(
    history_id:   uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    row = await app_state.db_pool.fetchrow(
        "SELECT * FROM v_history_feed WHERE history_id=$1 AND user_id=$2",
        history_id, current_user["id"],
    )
    if not row:
        raise HTTPException(404, "History item not found")
    return HistoryItem(**dict(row))


@router.patch("/{history_id}", response_model=MessageResponse, summary="Star / tag item")
async def update_history_item(
    history_id:   uuid.UUID,
    body:         UpdateHistoryRequest,
    current_user: dict = Depends(get_current_user),
):
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(400, "Nothing to update")

    updates = [f"{k}=${i+1}" for i, k in enumerate(fields)]
    params  = list(fields.values()) + [history_id, current_user["id"]]

    result = await app_state.db_pool.execute(
        f"UPDATE translation_history SET {', '.join(updates)} "
        f"WHERE id=${len(params)-1} AND user_id=${len(params)}",
        *params,
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "History item not found")
    return MessageResponse(message="Updated successfully")


@router.delete("/{history_id}", response_model=MessageResponse, summary="Delete history item")
async def delete_history_item(
    history_id:   uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    result = await app_state.db_pool.execute(
        "UPDATE translation_history SET is_deleted=TRUE WHERE id=$1 AND user_id=$2",
        history_id, current_user["id"],
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "History item not found")
    return MessageResponse(message="Deleted")


@router.delete("", response_model=MessageResponse, summary="Clear all history")
async def clear_history(current_user: dict = Depends(get_current_user)):
    await app_state.db_pool.execute(
        "UPDATE translation_history SET is_deleted=TRUE WHERE user_id=$1",
        current_user["id"],
    )
    return MessageResponse(message="History cleared")
