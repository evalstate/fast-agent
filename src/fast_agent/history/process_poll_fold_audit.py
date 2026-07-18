"""Typed audit archive for managed-process poll history folding."""

from pydantic import BaseModel, model_validator

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ArchivedPollExchange(BaseModel):
    """One exact poll request/result pair preserved outside effective history."""

    call_id: str
    request: PromptMessageExtended
    result: PromptMessageExtended

    @model_validator(mode="after")
    def validate_exchange(self) -> "ArchivedPollExchange":
        if set(self.request.tool_calls or {}) != {self.call_id}:
            raise ValueError("Archived poll request does not match its call ID")
        if set(self.result.tool_results or {}) != {self.call_id}:
            raise ValueError("Archived poll result does not match its call ID")
        return self


class ArchivedContextRewrite(BaseModel):
    """A historical rewrite of the model-visible prompt context."""

    after_call_id: str
    summary: str
    fold: dict[str, object]
    removed_call_ids: list[str]
    retained_call_ids: list[str]


class ProcessPollFoldAudit(BaseModel):
    """Lossless exchanges and context rewrites for one cumulative fold."""

    removed_exchanges: list[ArchivedPollExchange]
    retained_exchanges: list[ArchivedPollExchange]
    context_rewrites: list[ArchivedContextRewrite]

    @model_validator(mode="after")
    def validate_archive(self) -> "ProcessPollFoldAudit":
        if not self.retained_exchanges:
            raise ValueError("Poll fold audit must retain at least one exchange")
        if not self.context_rewrites:
            raise ValueError("Poll fold audit must contain a context rewrite")
        exchanges = [*self.removed_exchanges, *self.retained_exchanges]
        call_ids = [exchange.call_id for exchange in exchanges]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Poll fold audit contains duplicate call IDs")
        known_call_ids = set(call_ids)
        for rewrite in self.context_rewrites:
            referenced_call_ids = {
                rewrite.after_call_id,
                *rewrite.removed_call_ids,
                *rewrite.retained_call_ids,
            }
            if not referenced_call_ids <= known_call_ids:
                raise ValueError("Context rewrite references unknown call IDs")
        retained_call_ids = [
            exchange.call_id for exchange in self.retained_exchanges
        ]
        latest_rewrite = self.context_rewrites[-1]
        if (
            latest_rewrite.after_call_id != retained_call_ids[-1]
            or latest_rewrite.retained_call_ids != retained_call_ids
        ):
            raise ValueError("Latest context rewrite is not anchored to a retained call")
        return self
