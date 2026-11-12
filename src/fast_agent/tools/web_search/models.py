"""Wire models for the standalone Codex ``alpha/search`` endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

UInt64 = Annotated[int, Field(ge=0, le=2**64 - 1)]
SearchInput = str | list[dict[str, JsonValue]]
FinanceAssetType = Literal["equity", "fund", "crypto", "index"]
SportsFunction = Literal["schedule", "standings"]
SportsLeague = Literal["nba", "wnba", "nfl", "nhl", "mlb", "epl", "ncaamb", "ncaawb", "ipl"]
SearchResponseLength = Literal["short", "medium", "long"]
SearchContextSize = Literal["low", "medium", "high"]
AllowedCaller = Literal["direct", "shell", "code_interpreter"]
ExternalWebAccess = bool | Literal["cached", "indexed", "live"]


class SearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchQuery(SearchModel):
    q: str
    recency: UInt64 | None = Field(default=None, description="Filter to this many recent days.")
    domains: list[str] | None = None


class OpenOperation(SearchModel):
    ref_id: str = Field(description="Reference ID or URL to open.")
    lineno: UInt64 | None = None


class ClickOperation(SearchModel):
    ref_id: str = Field(description="Reference ID of the page containing the link.")
    id: UInt64 = Field(description="Numbered link ID.")


class FindOperation(SearchModel):
    ref_id: str
    pattern: str


class ScreenshotOperation(SearchModel):
    ref_id: str
    pageno: UInt64 = Field(description="Zero-indexed PDF page number.")


class FinanceOperation(SearchModel):
    ticker: str
    type: FinanceAssetType
    market: str | None = Field(
        default=None, description='Alpha-3 country code, OTC, or "" for crypto.'
    )


class WeatherOperation(SearchModel):
    location: str = Field(description="Country, Area, City.")
    start: str | None = Field(
        default=None, description="Start date (YYYY-MM-DD); defaults to today."
    )
    duration: UInt64 | None = None


class SportsOperation(SearchModel):
    # The live service requires this discriminator even though Codex makes it optional.
    tool: Literal["sports"] = "sports"
    fn: SportsFunction
    league: SportsLeague
    team: str | None = None
    opponent: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    num_games: UInt64 | None = None
    locale: str | None = None


class TimeOperation(SearchModel):
    utc_offset: str = Field(description='UTC offset, e.g. "+03:00".')


class SearchCommands(SearchModel):
    search_query: list[SearchQuery] | None = None
    image_query: list[SearchQuery] | None = None
    open: list[OpenOperation] | None = None
    click: list[ClickOperation] | None = None
    find: list[FindOperation] | None = None
    screenshot: list[ScreenshotOperation] | None = None
    finance: list[FinanceOperation] | None = None
    weather: list[WeatherOperation] | None = None
    sports: list[SportsOperation] | None = None
    time: list[TimeOperation] | None = None
    response_length: SearchResponseLength | None = None


class ApproximateLocation(SearchModel):
    type: Literal["approximate"] = "approximate"
    country: str | None = None
    region: str | None = None
    city: str | None = None
    timezone: str | None = None


class SearchFilters(SearchModel):
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None


class SearchImageSettings(SearchModel):
    max_results: UInt64 | None = None
    caption: bool | None = None


class SearchSettings(SearchModel):
    user_location: ApproximateLocation | None = None
    search_context_size: SearchContextSize | None = None
    filters: SearchFilters | None = None
    image_settings: SearchImageSettings | None = None
    allowed_callers: list[AllowedCaller] | None = None
    external_web_access: ExternalWebAccess | None = None


class SearchReasoning(SearchModel):
    effort: str | None = Field(
        default=None, description="Reasoning effort, including model-defined values."
    )
    summary: Literal["auto", "concise", "detailed", "none"] | None = None
    context: Literal["auto", "current_turn", "all_turns"] | None = None


class SearchRequest(SearchModel):
    id: str = Field(min_length=1, description="Caller-owned stable search session ID.")
    model: str = Field(min_length=1)
    reasoning: SearchReasoning | None = None
    input: SearchInput | None = None
    commands: SearchCommands | None = None
    settings: SearchSettings | None = None
    max_output_tokens: UInt64 | None = None


class SearchResponse(BaseModel):
    """Unmodified text and opaque, forward-compatible structured results."""

    model_config = ConfigDict(extra="allow", strict=True)

    encrypted_output: str | None = None
    output: str
    results: list[JsonValue] | None = None
