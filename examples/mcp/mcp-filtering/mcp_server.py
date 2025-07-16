import random
import string

from mcp.server.fastmcp import FastMCP

app = FastMCP(name="Creative Writing Server")

# String manipulation tools
@app.tool(
    name="reverse_string",
    description="Reverses a string",
)
def reverse_string(text: str) -> str:
    return text[::-1]

@app.tool(
    name="capitalize_string",
    description="Capitalizes a string",
)
def capitalize_string(text: str) -> str:
    return text.upper()

@app.tool(
    name="lowercase_string",
    description="Converts a string to lowercase",
)
def lowercase_string(text: str) -> str:
    return text.lower()

@app.tool(
    name="random_string",
    description="Generates a random string of a given length",
)
def random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@app.tool(
    name="random_case_string",
    description="Randomly capitalizes or lowercase each letter in a string",
)
def random_case_string(text: str) -> str:
    return ''.join(random.choice([str.upper, str.lower])(c) for c in text)

# Code formatting tools
@app.tool(
    name="coding_camel_case",
    description="Converts a string to camel case",
)
def coding_camel_case(text: str) -> str:
    return text.title().replace(" ", "")

@app.tool(
    name="coding_snake_case",
    description="Converts a string to snake case",
)
def coding_snake_case(text: str) -> str:
    return text.lower().replace(" ", "_")

@app.tool(
    name="coding_kebab_case",
    description="Converts a string to kebab case",
)
def coding_kebab_case(text: str) -> str:
    return text.lower().replace(" ", "-")

# Resources
@app.resource("resource://writing/style_guide")
def writing_style_guide() -> str:
    return """Writing Style Guide:
1. Use active voice when possible
2. Keep sentences concise and clear
3. Vary sentence structure for rhythm
4. Use strong, specific verbs
5. Avoid excessive adverbs"""

@app.resource("resource://writing/character_names")
def character_names() -> str:
    return """Character Name Ideas:
Fantasy: Eldara, Thorne, Zephyr, Lyanna
Modern: Alex, Jordan, Riley, Cameron
Historical: Eleanor, Benedict, Cordelia, Jasper
Sci-fi: Zara, Kai, Nova, Orion"""

@app.resource("resource://coding/conventions")
def coding_conventions() -> str:
    return """Coding Conventions:
- Variables: snake_case
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_CASE
- Files: lowercase with hyphens"""

@app.resource("resource://creativity/prompts")
def creativity_prompts() -> str:
    return """Creative Writing Prompts:
1. A character discovers they can hear colors
2. The last person on Earth receives a phone call
3. A library where books come to life at night
4. Time moves backwards for one day only
5. A world where lies become physical objects"""

# Prompts
@app.prompt("writing_assistant")
def writing_assistant(genre: str = "general") -> str:
    """Creative writing assistant for different genres"""
    return f"I am a creative writing assistant specialized in {genre} writing. I can help you with story structure, character development, dialogue, and prose improvement."

@app.prompt("writing_feedback")
def writing_feedback(focus: str = "overall") -> str:
    """Provides feedback on written work"""
    return f"I am a writing coach focused on {focus} feedback. I'll provide constructive criticism and suggestions to improve your writing."

@app.prompt("coding_helper")
def coding_helper(language: str = "python") -> str:
    """Coding assistant for formatting and conventions"""
    return f"I am a coding assistant specialized in {language}. I can help you with code formatting, naming conventions, and best practices."

@app.prompt("creative_brainstorm")
def creative_brainstorm(topic: str = "general") -> str:
    """Brainstorming assistant for creative projects"""
    return f"I am a creative brainstorming assistant focused on {topic}. I can help you generate ideas, explore concepts, and overcome creative blocks."



if __name__ == "__main__":
    # Run in stdio mode
    app.run()