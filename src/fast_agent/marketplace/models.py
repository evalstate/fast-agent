"""Shared marketplace pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MarketplaceEntryFieldsModel(BaseModel):
    name: str | None = None
    description: str | None = None
    kind: str | None = None
    repo_url: str | None = Field(default=None, alias="repo")
    repo_ref: str | None = None
    repo_path: str | None = None
    source_url: str | None = None
    bundle_name: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
