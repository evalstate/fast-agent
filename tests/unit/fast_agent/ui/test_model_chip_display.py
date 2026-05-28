from fast_agent.ui.model_chip_display import MODEL_CHIP_COLOR, render_model_chip

WEB_INDICATOR = "<style bg='ansigreen'>⊕</style>"
WEB_FETCH_INDICATOR = "<style bg='ansigreen'> ⇣</style>"
SERVICE_TIER_INDICATOR = "<style bg='ansired'>»</style>"
ATTACHMENT_INDICATOR = "<style bg='ansigreen'>▲2</style>"


def test_render_model_chip_places_indicators_after_model_label() -> None:
    chip = render_model_chip(
        model_label="gpt-5",
        web_search_indicator=WEB_INDICATOR,
        service_tier_indicator=SERVICE_TIER_INDICATOR,
        web_fetch_indicator=WEB_FETCH_INDICATOR,
        attachment_indicator=ATTACHMENT_INDICATOR,
    )

    assert chip == (
        f"<style bg='{MODEL_CHIP_COLOR}'>gpt-5</style>"
        f"{SERVICE_TIER_INDICATOR}{WEB_INDICATOR}{WEB_FETCH_INDICATOR}{ATTACHMENT_INDICATOR}"
    )


def test_render_model_chip_omits_missing_indicators() -> None:
    chip = render_model_chip(model_label="gpt-5")

    assert chip == f"<style bg='{MODEL_CHIP_COLOR}'>gpt-5</style>"


def test_render_model_chip_preserves_prefixed_model_label() -> None:
    chip = render_model_chip(model_label="∞gpt-5.3-codex")

    assert chip == f"<style bg='{MODEL_CHIP_COLOR}'>∞gpt-5.3-codex</style>"


def test_render_model_chip_preserves_overlay_prefixed_model_label() -> None:
    chip = render_model_chip(model_label="▼haikutiny")

    assert chip == f"<style bg='{MODEL_CHIP_COLOR}'>▼haikutiny</style>"
