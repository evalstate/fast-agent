#!/usr/bin/env python3
"""
Reasoning level gauge - comparing visualization options.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

BRAILLE_HEIGHTS = {1: "⣀", 2: "⣤", 3: "⣶", 4: "⣿"}
BRAILLE_FILL = {0: "⣿", 1: "⣀", 2: "⣤", 3: "⣶", 4: "⣿"}  # 0 shows full block
FULL = "⣿"
INACTIVE = "ansibrightblack"

level = 2
max_level = 4


def get_color(lvl: int) -> str:
    if lvl <= 1:
        return "ansigreen"
    elif lvl <= 2:
        return "ansiyellow"
    else:
        return "ansired"


def gauge_single(lvl: int, max_lvl: int = 4) -> str:
    """Single gauge - full block at rest (bright_black), fills up when active."""
    if lvl == 0:
        return f"<style bg='{INACTIVE}'>{FULL}</style>"
    char = BRAILLE_FILL.get(min(lvl, max_lvl), BRAILLE_FILL[max_lvl])
    return f"<style bg='ansiyellow'>{char}</style>"


def gauge_stair2(lvl: int, max_lvl: int = 4) -> str:
    """Staircase - inactive = bright_black braille."""
    active_color = get_color(lvl)
    
    parts = []
    for i in range(1, max_lvl + 1):
        char = BRAILLE_HEIGHTS.get(i, "⣿")
        if i <= lvl:
            parts.append(f"<style bg='{active_color}'>{char}</style>")
        else:
            parts.append(f"<style bg='{INACTIVE}'>{char}</style>")
    
    return "".join(parts)


def gauge_block2(lvl: int, max_lvl: int = 4) -> str:
    """Full blocks - inactive = bright_black braille."""
    active_color = get_color(lvl)
    
    parts = []
    for i in range(1, max_lvl + 1):
        if i <= lvl:
            parts.append(f"<style bg='{active_color}'>{FULL}</style>")
        else:
            parts.append(f"<style bg='{INACTIVE}'>{FULL}</style>")
    
    return "".join(parts)


def get_toolbar():
    return HTML(
        f" STAIR2: {gauge_stair2(level, max_level)}"
        f" | BLOCK2: {gauge_block2(level, max_level)}"
        f" | SINGLE: {gauge_single(level, max_level)}"
        f" | L:{level}"
    )


def main():
    print("\n" + "="*70)
    print("Reasoning Gauge Options")
    print("="*70)
    print("""
Comparing:

  STAIR2: ⣀⣤⣶⣿  (inactive = bright_black braille)
  BLOCK2: ⣿⣿⣿⣿  (inactive = bright_black braille)
  SINGLE: ⣿       (bright_black full block at rest)

Active: green → yellow → red

Commands: +/- (level), q (quit)
""")
    
    style = Style.from_dict({
        "bottom-toolbar": "#ansiblack bg:#ansigray",
    })
    
    session = PromptSession(
        bottom_toolbar=get_toolbar,
        style=style,
        erase_when_done=False,
    )
    
    global level
    
    while True:
        try:
            text = session.prompt("> ").strip().lower()
            if text == "q":
                break
            elif text == "+":
                level = min(level + 1, max_level)
            elif text == "-":
                level = max(level - 1, 0)
        except (KeyboardInterrupt, EOFError):
            break
    
    print("Done!")


if __name__ == "__main__":
    main()
