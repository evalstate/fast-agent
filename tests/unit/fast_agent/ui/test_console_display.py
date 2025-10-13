"""Unit tests for console_display module, specifically _prepare_markdown_content."""

from fast_agent.ui.console_display import _prepare_markdown_content


class TestPrepareMarkdownContent:
    """Test the _prepare_markdown_content function."""

    def test_none_input(self):
        """Test that None input returns None unchanged."""
        result = _prepare_markdown_content(None)
        assert result is None

    def test_none_input_with_escape_false(self):
        """Test that None input returns None even when escape_xml is False."""
        result = _prepare_markdown_content(None, escape_xml=False)
        assert result is None

    def test_empty_string(self):
        """Test that empty string doesn't crash and returns empty string."""
        result = _prepare_markdown_content("")
        assert result == ""

    def test_empty_string_with_escape_false(self):
        """Test that empty string returns empty when escape_xml is False."""
        result = _prepare_markdown_content("", escape_xml=False)
        assert result == ""

    def test_escape_xml_false_returns_unchanged(self):
        """Test that escape_xml=False returns content unchanged."""
        content = "<tag>content & 'quotes' \"double\"</tag>"
        result = _prepare_markdown_content(content, escape_xml=False)
        assert result == content

    def test_non_string_input(self):
        """Test that non-string inputs are returned unchanged."""
        # Test with integer
        result = _prepare_markdown_content(123)
        assert result == 123

        # Test with list
        result = _prepare_markdown_content([1, 2, 3])
        assert result == [1, 2, 3]

        # Test with dict
        test_dict = {"key": "value"}
        result = _prepare_markdown_content(test_dict)
        assert result == test_dict

    def test_basic_html_escaping(self):
        """Test that HTML characters are properly escaped outside code blocks."""
        content = "This has <tag> and & and > and < and \" and ' characters"
        result = _prepare_markdown_content(content)
        expected = (
            "This has &lt;tag&gt; and &amp; and &gt; and &lt; and &quot; and &#39; characters"
        )
        assert result == expected

    def test_preserves_fenced_code_blocks(self):
        """Test that content inside fenced code blocks is not escaped."""
        content = """Before code
```python
def func():
    return "<tag>" & 'value'
```
After code with <tag>"""
        result = _prepare_markdown_content(content)

        # Check that code block content is preserved
        assert "def func():" in result
        assert "return \"<tag>\" & 'value'" in result

        # Check that content outside code blocks is escaped
        assert "After code with &lt;tag&gt;" in result

    def test_preserves_inline_code(self):
        """Test that content inside inline code is not escaped."""
        content = "Use `<tag>` and `x & y` in code, but escape <tag> outside"
        result = _prepare_markdown_content(content)

        # Inline code should be preserved
        assert "`<tag>`" in result
        assert "`x & y`" in result

        # Outside content should be escaped
        assert "but escape &lt;tag&gt; outside" in result

    def test_multiple_code_blocks(self):
        """Test handling of multiple code blocks in the same content."""
        content = """First <tag>
```
<code1> & "quotes"
```
Middle <tag>
```
<code2> & 'quotes'
```
End <tag>"""
        result = _prepare_markdown_content(content)

        # Code blocks should be preserved
        assert '<code1> & "quotes"' in result
        assert "<code2> & 'quotes'" in result

        # Outside content should be escaped
        assert "First &lt;tag&gt;" in result
        assert "Middle &lt;tag&gt;" in result
        assert "End &lt;tag&gt;" in result

    def test_mixed_inline_and_fenced_code(self):
        """Test content with both inline and fenced code blocks."""
        content = """Use `<inline>` here
```
<fenced> & "code"
```
And `<more>` inline with <tag> outside"""
        result = _prepare_markdown_content(content)

        # Both types of code should be preserved
        assert "`<inline>`" in result
        assert '<fenced> & "code"' in result
        assert "`<more>`" in result

        # Outside content should be escaped
        assert "with &lt;tag&gt; outside" in result

    def test_empty_code_blocks(self):
        """Test that empty code blocks don't cause issues."""
        content = """Before
```
```
After <tag>"""
        result = _prepare_markdown_content(content)
        assert "After &lt;tag&gt;" in result

    def test_nested_backticks_not_treated_as_inline_code(self):
        """Test that triple backticks are not treated as inline code."""
        content = "This ```is not``` inline code <tag>"
        result = _prepare_markdown_content(content)
        # The content between triple backticks should be escaped
        assert "```is not``` inline code &lt;tag&gt;" in result

    def test_single_backtick_not_treated_as_code(self):
        """Test that single backtick without closing is not treated as code."""
        content = "This ` is not code <tag>"
        result = _prepare_markdown_content(content)
        assert "This ` is not code &lt;tag&gt;" in result

    def test_all_escape_characters(self):
        """Test that all defined escape characters are properly replaced."""
        content = "& < > \" '"
        result = _prepare_markdown_content(content)
        assert result == "&amp; &lt; &gt; &quot; &#39;"

    def test_preserve_newlines_and_whitespace(self):
        """Test that newlines and whitespace are preserved."""
        content = "Line 1\n  Line 2 with spaces\n\tLine 3 with tab"
        result = _prepare_markdown_content(content)
        assert "Line 1\n  Line 2 with spaces\n\tLine 3 with tab" == result

    def test_code_block_at_start(self):
        """Test code block at the very start of content."""
        content = """```
<code>
```
After <tag>"""
        result = _prepare_markdown_content(content)
        assert "<code>" in result
        assert "After &lt;tag&gt;" in result

    def test_code_block_at_end(self):
        """Test code block at the very end of content."""
        content = """Before <tag>
```
<code>
```"""
        result = _prepare_markdown_content(content)
        assert "Before &lt;tag&gt;" in result
        assert "<code>" in result

    def test_adjacent_inline_code(self):
        """Test adjacent inline code blocks.

        With the markdown-it parser approach, adjacent inline code is properly
        identified and preserved (unlike the old regex approach).
        """
        # Test with space between inline code blocks (works correctly)
        content = "`<code1>` `<code2>` and <tag>"
        result = _prepare_markdown_content(content)
        assert "`<code1>`" in result
        assert "`<code2>`" in result
        assert "and &lt;tag&gt;" in result

        # Adjacent without space - markdown-it parser handles this correctly
        content_adjacent = "`<code1>``<code2>` and <tag>"
        result_adjacent = _prepare_markdown_content(content_adjacent)
        # With parser-based approach, inline code is properly identified and preserved
        assert "`<code1>`" in result_adjacent
        assert "`<code2>`" in result_adjacent
        assert "and &lt;tag&gt;" in result_adjacent

    def test_realistic_xml_content(self):
        """Test with realistic XML content that should be escaped."""
        content = """Here's an XML example:
<root>
    <child attr="value">Content & more</child>
</root>
But in code it's preserved:
```xml
<root>
    <child attr="value">Content & more</child>
</root>
```"""
        result = _prepare_markdown_content(content)

        # Outside code should be escaped
        assert "&lt;root&gt;" in result
        assert "&lt;child attr=&quot;value&quot;&gt;" in result

        # Inside code should be preserved
        assert '    <child attr="value">Content & more</child>' in result

    def test_multiple_code_blocks_with_language_tags(self):
        """Test handling of multiple code blocks with language specifiers (e.g., ```html, ```typescript)."""
        content = """I'll create a compact, self-contained TypeScript game (~200 lines) that runs in the browser.
It's a mini "Dodge-Faller" arcade: you move left/right with arrow keys, avoid falling blocks, and survive as long as possible. Everything (HTML, CSS, TS) is in one snippet so you can copy-paste it into a single `.ts` file, compile with `tsc`, and open the resulting `.html`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Dodge-Faller</title>
  <style>
    body { margin: 0; background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; }
    canvas { background: #000; border: 2px solid #fff; }
  </style>
</head>
<body>
  <canvas id="c" width="400" height="600"></canvas>
  <script src="game.js"></script>
</body>
</html>
```

```typescript
// game.ts
const canvas = document.getElementById("c") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const W = canvas.width;
const H = canvas.height;

const PLAYER_W = 40;
const PLAYER_H = 20;
const PLAYER_SPEED = 6;
const BLOCK_W = 30;
const BLOCK_H = 30;
const BLOCK_SPEED = 3;
const BLOCK_SPAWN_EVERY = 45; // frames

let playerX = W / 2 - PLAYER_W / 2;
let playerY = H - PLAYER_H - 10;
let blocks: { x: number; y: number }[] = [];
let frame = 0;
let running = true;
let score = 0;

const keys: Record<string, boolean> = {};
window.addEventListener("keydown", e => keys[e.key] = true);
window.addEventListener("keyup", e => keys[e.key] = false);

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function collides(ax: number, ay: number, aw: number, ah: number,
              bx: number, by: number, bw: number, bh: number) {
  return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

function update() {
  if (!running) return;
  frame++;

  // move player
  if (keys["ArrowLeft"]) playerX = Math.max(0, playerX - PLAYER_SPEED);
  if (keys["ArrowRight"]) playerX = Math.min(W - PLAYER_W, playerX + PLAYER_SPEED);

  // spawn blocks
  if (frame % BLOCK_SPAWN_EVERY === 0) {
    blocks.push({ x: rand(0, W - BLOCK_W), y: -BLOCK_H });
  }

  // move blocks
  for (const b of blocks) b.y += BLOCK_SPEED;

  // remove off-screen blocks
  blocks = blocks.filter(b => b.y < H + BLOCK_H);

  // collisions
  for (const b of blocks) {
    if (collides(playerX, playerY, PLAYER_W, PLAYER_H, b.x, b.y, BLOCK_W, BLOCK_H)) {
      running = false;
    }
  }

  score = Math.max(score, Math.floor(frame / 10));
}

function draw() {
  ctx.clearRect(0, 0, W, H);

  // player
  ctx.fillStyle = "#0af";
  ctx.fillRect(playerX, playerY, PLAYER_W, PLAYER_H);

  // blocks
  ctx.fillStyle = "#f44";
  for (const b of blocks) ctx.fillRect(b.x, b.y, BLOCK_W, BLOCK_H);

  // score
  ctx.fillStyle = "#fff";
  ctx.font = "20px monospace";
  ctx.fillText(`Score: ${score}`, 10, 30);

  if (!running) {
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.font = "30px monospace";
    ctx.fillText("Game Over", W / 2 - 70, H / 2);
    ctx.font = "16px monospace";
    ctx.fillText("Refresh to play again", W / 2 - 90, H / 2 + 30);
  }
}

function loop() {
  update();
  draw();
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
```

Compile with
`tsc game.ts --target ES2020 --module none --outFile game.js`
and open the HTML file in your browser."""

        result = _prepare_markdown_content(content)

        # Debug: print the result to see what's happening
        print("\n=== RESULT ===")
        print(result)
        print("\n=== END RESULT ===")

        # The code blocks should be preserved exactly as they are (no HTML encoding inside)
        assert '<meta charset="utf-8" />' in result, "HTML code block content should not be escaped"
        assert 'ctx.fillText("Game Over", W / 2 - 70, H / 2)' in result, "TypeScript string literals should not be escaped"

        # Content should not be duplicated
        # Note: "Game Over" appears once, but requestAnimationFrame(loop) appears twice
        # (once inside loop(), once to call it initially)
        assert result.count('ctx.fillText("Game Over"') == 1, "Content should not be duplicated"
        assert result.count('requestAnimationFrame(loop)') == 2, "Should match the 2 instances in original code (not 4 like the bug produced)"

        # Outside code blocks, backticks for inline code should be preserved
        assert "`tsc game.ts" in result or "tsc game.ts" in result, "Inline code or plain text should be present"
