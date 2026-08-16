"""Provisional local wire models for SEP-2640 Draft d7490ecd.

The pinned MCP SDK does not yet provide Skills Extension request/result types.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp_types import (
    CacheableResult,
    PaginatedRequestParams,
    PaginatedResult,
    Request,
    RequestParams,
    Result,
)
from pydantic import BaseModel, ConfigDict, Field


class SkillResource(BaseModel):
    """A content-addressed resource belonging to a skill."""

    model_config = ConfigDict(populate_by_name=True)

    uri: str = Field(alias="uri")
    digest: str = Field(alias="digest")


class SkillEntry(BaseModel):
    """A skill's metadata and optional complete resource manifest."""

    model_config = ConfigDict(populate_by_name=True)

    uri: str = Field(alias="uri")
    frontmatter: dict[str, Any] = Field(alias="frontmatter")
    resources: list[SkillResource] | None = Field(default=None, alias="resources")


class ListSkillsRequestParams(PaginatedRequestParams):
    """Parameters for the ``skills/list`` request."""


class ListSkillsRequest(Request[ListSkillsRequestParams, Literal["skills/list"]]):
    """Request for the SEP-2640 skills listing."""

    method: Literal["skills/list"] = "skills/list"
    params: ListSkillsRequestParams


class ListSkillsResult(PaginatedResult, CacheableResult):
    """The paginated, cacheable response to ``skills/list``."""

    skills: list[SkillEntry] = Field(alias="skills")
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class GetSkillRequestParams(RequestParams):
    """Parameters for the ``skills/get`` request."""

    uri: str = Field(alias="uri")


class GetSkillRequest(Request[GetSkillRequestParams, Literal["skills/get"]]):
    """Request for a single SEP-2640 skill entry."""

    method: Literal["skills/get"] = "skills/get"
    params: GetSkillRequestParams


class GetSkillResult(Result):
    """The response to ``skills/get``."""

    skill: SkillEntry = Field(alias="skill")
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class DirectoryReadRequestParams(PaginatedRequestParams):
    """Parameters for the SEP-2640 directory-read extension."""

    uri: str = Field(alias="uri")


class DirectoryReadRequest(
    Request[DirectoryReadRequestParams, Literal["resources/directory/read"]]
):
    """Request for the SEP-2640 directory-read extension."""

    method: Literal["resources/directory/read"] = "resources/directory/read"
    params: DirectoryReadRequestParams
