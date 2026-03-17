"""
schemas/models.py — Pydantic v2 request / response schemas
"""

from __future__ import annotations
from datetime   import datetime
from typing     import Optional, List, Literal
from uuid       import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ══════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name:     str        = Field(..., min_length=1, max_length=100)
    email:    EmailStr
    password: str        = Field(..., min_length=8, max_length=128)

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserResponse


# ══════════════════════════════════════════════════════════════════════
#  USERS
# ══════════════════════════════════════════════════════════════════════

class UserResponse(BaseModel):
    id:         UUID
    name:       str
    email:      str
    plan:       str
    created_at: datetime

class UpdateProfileRequest(BaseModel):
    name:       Optional[str] = Field(None, min_length=1, max_length=100)
    avatar_url: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════

class UserSettingsResponse(BaseModel):
    source_language_code:  str
    target_language_code:  str
    asr_model_id:          Optional[int]
    mt_model_id:           Optional[int]
    tts_model_id:          Optional[int]
    auto_translate:        bool
    translate_on_stop:     bool
    auto_play_audio:       bool
    save_history:          bool
    noise_cancellation:    bool
    haptic_feedback:       bool
    theme:                 str
    default_input_mode:    str

class UpdateSettingsRequest(BaseModel):
    source_language_code:  Optional[str] = None
    target_language_code:  Optional[str] = None
    asr_model_id:          Optional[int] = None
    mt_model_id:           Optional[int] = None
    tts_model_id:          Optional[int] = None
    auto_translate:        Optional[bool] = None
    translate_on_stop:     Optional[bool] = None
    auto_play_audio:       Optional[bool] = None
    save_history:          Optional[bool] = None
    noise_cancellation:    Optional[bool] = None
    haptic_feedback:       Optional[bool] = None
    theme:                 Optional[Literal["light","dark","system"]] = None
    default_input_mode:    Optional[Literal["speech","text","file"]] = None


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION
# ══════════════════════════════════════════════════════════════════════

class TranslateTextRequest(BaseModel):
    text:            str  = Field(..., min_length=1, max_length=4000,
                                  description="English text to translate")
    target_language: Literal["fr", "tw"] = Field(
        "fr", description="Target language: fr=French, tw=Twi"
    )
    generate_audio:  bool = Field(True,  description="Also run TTS to produce audio")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

class TranslationResponse(BaseModel):
    session_id:        UUID
    source_language:   str
    target_language:   str
    asr_transcript:    str
    asr_confidence:    Optional[float]
    translated_text:   str
    audio_url:         Optional[str]    = None   # served from /audio/<filename>
    asr_duration_ms:   int
    mt_duration_ms:    int
    tts_duration_ms:   int
    total_duration_ms: int
    status:            str

class TranscribeResponse(BaseModel):
    """Response for /translate/transcribe — ASR only, no MT/TTS."""
    transcript:  str
    confidence:  float
    duration_ms: int


# ══════════════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════════════

class HistoryItem(BaseModel):
    history_id:        UUID
    session_id:        UUID
    is_starred:        bool
    tags:              Optional[List[str]]
    source_language:   str
    source_flag:       Optional[str]
    target_language:   str
    target_flag:       Optional[str]
    source_text:       Optional[str]
    translated_text:   Optional[str]
    audio_url:         Optional[str]
    asr_confidence:    Optional[float]
    asr_model_used:    Optional[str]
    mt_model_used:     Optional[str]
    tts_model_used:    Optional[str]
    total_duration_ms: Optional[int]
    status:            str
    translated_at:     datetime

class HistoryListResponse(BaseModel):
    items:  List[HistoryItem]
    total:  int
    limit:  int
    offset: int

class UpdateHistoryRequest(BaseModel):
    is_starred: Optional[bool]      = None
    tags:       Optional[List[str]] = None


# ══════════════════════════════════════════════════════════════════════
#  PHRASEBOOK
# ══════════════════════════════════════════════════════════════════════

class PhraseResponse(BaseModel):
    id:                  int
    category_id:         int
    category_name:       str
    category_icon:       Optional[str]
    source_language:     str
    target_language:     str
    source_text:         str
    translated_text:     str
    audio_url:           Optional[str]

class PhraseCategoryResponse(BaseModel):
    id:         int
    name:       str
    icon:       Optional[str]
    sort_order: int
    phrases:    List[PhraseResponse] = []

class SavePhraseRequest(BaseModel):
    phrase_id:           Optional[int]  = None   # system phrase
    custom_source:       Optional[str]  = None   # user custom phrase
    custom_target:       Optional[str]  = None
    source_language_code: Optional[str] = None
    target_language_code: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
#  MODELS REGISTRY
# ══════════════════════════════════════════════════════════════════════

class ASRModelResponse(BaseModel):
    id:             int
    hf_model_id:    str
    short_name:     str
    description:    Optional[str]
    language_code:  str
    architecture:   str
    model_size:     Optional[str]
    is_downloaded:  bool
    is_active:      bool
    word_error_rate: Optional[float]

class MTModelResponse(BaseModel):
    id:              int
    hf_model_id:     str
    short_name:      str
    description:     Optional[str]
    source_language: str
    target_language: str
    architecture:    str
    is_downloaded:   bool
    is_active:       bool
    bleu_score:      Optional[float]

class TTSModelResponse(BaseModel):
    id:            int
    hf_model_id:   str
    short_name:    str
    description:   Optional[str]
    language_code: str
    architecture:  str
    sampling_rate: Optional[int]
    is_downloaded: bool
    is_active:     bool

class ModelRegistryResponse(BaseModel):
    asr: List[ASRModelResponse]
    mt:  List[MTModelResponse]
    tts: List[TTSModelResponse]


# ══════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════

class UserStatsResponse(BaseModel):
    total_translations:     int
    total_words_translated: int
    total_audio_seconds:    int
    avg_asr_confidence:     Optional[float]
    most_used_target_lang:  Optional[str]
    streak_days:            int
    last_active_at:         Optional[datetime]


# ══════════════════════════════════════════════════════════════════════
#  GENERIC
# ══════════════════════════════════════════════════════════════════════

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error:  str
    detail: Optional[str] = None
