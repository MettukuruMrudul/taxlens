"""Request bodies for the API (responses are plain JSON dicts)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    tax_year: int | None = Field(None, ge=2000, le=2100,
                                 description="Tax year chosen in the UI; overrides detection")


class FeedbackRequest(BaseModel):
    query_id: int
    rating: Literal[-1, 1]
    comment: str | None = Field(None, max_length=2000)


class TokenRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
