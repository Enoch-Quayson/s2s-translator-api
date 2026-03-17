"""
routers/phrasebook.py — phrasebook endpoints

Routes:
  GET  /phrasebook                   All categories + phrases
  GET  /phrasebook/categories        Category list
  GET  /phrasebook/phrases           Phrases (filterable by category / language)
  POST /phrasebook/saved             Save a phrase
  GET  /phrasebook/saved             User's saved phrases
  DELETE /phrasebook/saved/{id}      Remove saved phrase
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from core.state  import app_state
from core.auth   import get_current_user
from schemas.models import (
    PhraseCategoryResponse, PhraseResponse,
    SavePhraseRequest, MessageResponse,
)

router = APIRouter()
log    = logging.getLogger("s2s.phrasebook")


@router.get("/categories", response_model=list[PhraseCategoryResponse],
            summary="List all phrase categories")
async def list_categories():
    rows = await app_state.db_pool.fetch(
        "SELECT id, name, icon, sort_order FROM phrase_categories ORDER BY sort_order"
    )
    return [PhraseCategoryResponse(**dict(r), phrases=[]) for r in rows]


@router.get("", response_model=list[PhraseCategoryResponse],
            summary="All categories with their phrases")
async def get_phrasebook(
    source_lang: str           = Query("en"),
    target_lang: str           = Query("fr"),
):
    cat_rows = await app_state.db_pool.fetch(
        "SELECT id, name, icon, sort_order FROM phrase_categories ORDER BY sort_order"
    )

    phrase_rows = await app_state.db_pool.fetch("""
        SELECT p.id, p.category_id, pc.name AS category_name, pc.icon AS category_icon,
               p.source_language_code AS source_language,
               p.target_language_code AS target_language,
               p.source_text, p.translated_text, p.audio_url
        FROM phrases p
        JOIN phrase_categories pc ON pc.id = p.category_id
        WHERE p.source_language_code=$1 AND p.target_language_code=$2
        ORDER BY p.id
    """, source_lang, target_lang)

    phrase_map: dict[int, list] = {}
    for pr in phrase_rows:
        cid = pr["category_id"]
        phrase_map.setdefault(cid, []).append(PhraseResponse(**dict(pr)))

    return [
        PhraseCategoryResponse(**dict(c), phrases=phrase_map.get(c["id"], []))
        for c in cat_rows
    ]


@router.get("/phrases", response_model=list[PhraseResponse],
            summary="Get phrases (filterable)")
async def list_phrases(
    category_id: Optional[int] = Query(None),
    source_lang: str           = Query("en"),
    target_lang: str           = Query("fr"),
    q:           Optional[str] = Query(None, description="Search in source text"),
):
    conditions = ["p.source_language_code=$1", "p.target_language_code=$2"]
    params: list = [source_lang, target_lang]
    i = 3

    if category_id:
        conditions.append(f"p.category_id=${i}"); params.append(category_id); i+=1
    if q:
        conditions.append(f"p.source_text ILIKE ${i}"); params.append(f"%{q}%"); i+=1

    rows = await app_state.db_pool.fetch(f"""
        SELECT p.id, p.category_id, pc.name AS category_name, pc.icon AS category_icon,
               p.source_language_code AS source_language,
               p.target_language_code AS target_language,
               p.source_text, p.translated_text, p.audio_url
        FROM phrases p
        JOIN phrase_categories pc ON pc.id = p.category_id
        WHERE {" AND ".join(conditions)}
        ORDER BY p.id
    """, *params)

    return [PhraseResponse(**dict(r)) for r in rows]


@router.post("/saved", response_model=MessageResponse, status_code=201,
             summary="Save a phrase to user's phrasebook")
async def save_phrase(
    body:         SavePhraseRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.phrase_id and not (body.custom_source and body.custom_target):
        raise HTTPException(400, "Provide either phrase_id or custom_source + custom_target")

    try:
        await app_state.db_pool.execute("""
            INSERT INTO user_saved_phrases
                (user_id, phrase_id, custom_source, custom_target,
                 source_language_code, target_language_code)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (user_id, phrase_id) DO NOTHING
        """,
            current_user["id"],
            body.phrase_id,
            body.custom_source,
            body.custom_target,
            body.source_language_code or "en",
            body.target_language_code or "fr",
        )
    except Exception as e:
        raise HTTPException(400, str(e))

    return MessageResponse(message="Phrase saved")


@router.get("/saved", response_model=list[PhraseResponse],
            summary="Get user's saved phrases")
async def get_saved_phrases(
    current_user: dict = Depends(get_current_user),
):
    rows = await app_state.db_pool.fetch("""
        SELECT
            COALESCE(p.id, usp.id::int)                     AS id,
            COALESCE(p.category_id, 0)                      AS category_id,
            COALESCE(pc.name, 'Custom')                     AS category_name,
            COALESCE(pc.icon, '📌')                          AS category_icon,
            COALESCE(p.source_language_code, usp.source_language_code) AS source_language,
            COALESCE(p.target_language_code, usp.target_language_code) AS target_language,
            COALESCE(p.source_text,    usp.custom_source)   AS source_text,
            COALESCE(p.translated_text,usp.custom_target)   AS translated_text,
            p.audio_url
        FROM user_saved_phrases usp
        LEFT JOIN phrases p ON p.id = usp.phrase_id
        LEFT JOIN phrase_categories pc ON pc.id = p.category_id
        WHERE usp.user_id = $1
        ORDER BY usp.created_at DESC
    """, current_user["id"])

    return [PhraseResponse(**dict(r)) for r in rows]


@router.delete("/saved/{saved_id}", response_model=MessageResponse,
               summary="Remove a saved phrase")
async def delete_saved_phrase(
    saved_id:     uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    result = await app_state.db_pool.execute(
        "DELETE FROM user_saved_phrases WHERE id=$1 AND user_id=$2",
        saved_id, current_user["id"],
    )
    if result == "DELETE 0":
        raise HTTPException(404, "Saved phrase not found")
    return MessageResponse(message="Removed")
