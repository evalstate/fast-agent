"""Unit tests for mermaid_utils module."""

from fast_agent.ui.mermaid_utils import (
    MermaidDiagram,
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
    format_mermaid_links,
)


class TestExtractMermaidDiagrams:
    """Test diagram extraction from text content."""

    def test_no_diagrams(self):
        """Test extraction when no diagrams are present."""
        text = "This is just regular text without any diagrams."
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 0

    def test_single_diagram_no_title(self):
        """Test extraction of a single diagram without title."""
        text = """Some text before
```mermaid
graph TD
    A[Start] --> B[End]
```
Some text after"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 1
        assert diagrams[0].content == "graph TD\n    A[Start] --> B[End]"
        assert diagrams[0].title is None

    def test_single_diagram_with_title(self):
        """Test extraction of a single diagram with title."""
        text = """Some text before
```mermaid title={Flow Chart}
graph TD
    A[Start] --> B[End]
```
Some text after"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 1
        assert diagrams[0].content == "graph TD\n    A[Start] --> B[End]"
        assert diagrams[0].title == "Flow Chart"

    def test_multiple_diagrams(self):
        """Test extraction of multiple diagrams."""
        text = """First diagram:
```mermaid
graph LR
    A --> B
```

Second diagram with title:
```mermaid title={Sequence Diagram}
sequenceDiagram
    Alice->>Bob: Hello
```

Third diagram:
```mermaid
pie title Pets
    "Dogs" : 386
    "Cats" : 85
```"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 3

        assert diagrams[0].content == "graph LR\n    A --> B"
        assert diagrams[0].title is None

        assert diagrams[1].content == "sequenceDiagram\n    Alice->>Bob: Hello"
        assert diagrams[1].title == "Sequence Diagram"

        assert diagrams[2].content == 'pie title Pets\n    "Dogs" : 386\n    "Cats" : 85'
        assert diagrams[2].title == "Pets"

    def test_empty_diagram_block(self):
        """Test that empty diagram blocks are ignored."""
        text = """```mermaid
```
```mermaid title={Empty}

```"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 0

    def test_diagram_with_special_characters(self):
        """Test diagram with special characters in content."""
        text = """```mermaid
graph TD
    A["Node with quotes"] --> B{Decision?}
    B -->|Yes| C[/Parallelogram/]
    B -->|No| D((Circle))
```"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 1
        assert '"Node with quotes"' in diagrams[0].content
        assert "{Decision?}" in diagrams[0].content

    def test_title_extraction_from_content(self):
        """Test extraction of title from diagram content."""
        # Generic title
        text1 = """```mermaid
title User Authentication Flow
graph TD
    A[User] --> B[Login]
```"""
        diagrams1 = extract_mermaid_diagrams(text1)
        assert len(diagrams1) == 1
        assert diagrams1[0].title == "User Authentication Flow"

        # Pie chart title
        text2 = """```mermaid
pie title Monthly Sales Distribution
    "Product A" : 45
    "Product B" : 35
    "Product C" : 20
```"""
        diagrams2 = extract_mermaid_diagrams(text2)
        assert len(diagrams2) == 1
        assert diagrams2[0].title == "Monthly Sales Distribution"

        # Gantt chart title
        text3 = """```mermaid
gantt title Project Timeline
    section Phase 1
    Task 1 :a1, 2024-01-01, 30d
```"""
        diagrams3 = extract_mermaid_diagrams(text3)
        assert len(diagrams3) == 1
        assert diagrams3[0].title == "Project Timeline"

    def test_title_priority(self):
        """Test that code fence title takes priority over content title."""
        text = """```mermaid title={External Title}
title Internal Title
graph TD
    A --> B
```"""
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 1
        assert diagrams[0].title == "External Title"


class TestCreateMermaidLiveLink:
    """Test Mermaid Live link generation."""

    def test_simple_diagram(self):
        """Test link generation for a simple diagram."""
        diagram = "graph TD\n    A --> B"
        link = create_mermaid_live_link(diagram)
        assert link.startswith("https://www.mermaidchart.com/play#pako:")
        assert len(link) > 40  # Should have substantial encoded content

    def test_diagram_with_quotes(self):
        """Test link generation with quotes in diagram."""
        diagram = 'graph TD\n    A["Node with quotes"] --> B'
        link = create_mermaid_live_link(diagram)
        assert link.startswith("https://www.mermaidchart.com/play#pako:")
        # Should properly escape quotes

    def test_multiline_diagram(self):
        """Test link generation for multiline diagram."""
        diagram = """sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!"""
        link = create_mermaid_live_link(diagram)
        assert link.startswith("https://www.mermaidchart.com/play#pako:")


class TestFormatMermaidLinks:
    """Test markdown link formatting."""

    def test_single_diagram_no_title(self):
        """Test formatting single diagram without title."""
        diagrams = [MermaidDiagram(content="graph TD\n    A --> B")]
        links = format_mermaid_links(diagrams)
        assert len(links) == 1
        assert links[0].startswith("Diagram 1: [Open Diagram](")
        assert "pako:" in links[0]

    def test_single_diagram_with_title(self):
        """Test formatting single diagram with title."""
        diagrams = [MermaidDiagram(content="graph TD\n    A --> B", title="My Flow")]
        links = format_mermaid_links(diagrams)
        assert len(links) == 1
        assert links[0].startswith("Diagram 1 - My Flow: [Open Diagram](")

    def test_multiple_diagrams_mixed(self):
        """Test formatting multiple diagrams with mixed titles."""
        diagrams = [
            MermaidDiagram(content="graph TD\n    A --> B"),
            MermaidDiagram(content="pie\n    'A': 50", title="Statistics"),
            MermaidDiagram(content="graph LR\n    X --> Y"),
        ]
        links = format_mermaid_links(diagrams)
        assert len(links) == 3
        assert links[0].startswith("Diagram 1: [Open Diagram](")
        assert links[1].startswith("Diagram 2 - Statistics: [Open Diagram](")
        assert links[2].startswith("Diagram 3: [Open Diagram](")

    def test_empty_list(self):
        """Test formatting empty diagram list."""
        links = format_mermaid_links([])
        assert len(links) == 0


class TestIntegration:
    """Integration tests combining extraction and formatting."""

    def test_full_workflow(self):
        """Test complete workflow from text to formatted links."""
        text = """Here's a simple flow:
```mermaid title={User Flow}
graph TD
    A[User] --> B[Login]
    B --> C{Authenticated?}
    C -->|Yes| D[Dashboard]
    C -->|No| E[Error]
```

And a sequence:
```mermaid
sequenceDiagram
    Alice->>Bob: Request
    Bob-->>Alice: Response
```"""

        # Extract diagrams
        diagrams = extract_mermaid_diagrams(text)
        assert len(diagrams) == 2

        # Format links
        links = format_mermaid_links(diagrams)
        assert len(links) == 2
        assert "Diagram 1 - User Flow:" in links[0]
        assert "Diagram 2:" in links[1]

        # Verify links are valid
        for link in links:
            assert "[Open Diagram](" in link
            assert "https://www.mermaidchart.com/play#pako:" in link


class TestDetectDiagramType:
    """Test diagram type detection."""

    def test_flowchart_detection(self):
        """Test detection of flowchart diagrams."""
        assert detect_diagram_type("graph TD\n    A --> B") == "Flowchart"
        assert detect_diagram_type("flowchart LR\n    Start --> End") == "Flowchart"

    def test_sequence_detection(self):
        """Test detection of sequence diagrams."""
        assert detect_diagram_type("sequenceDiagram\n    Alice->>Bob: Hello") == "Sequence"

    def test_pie_chart_detection(self):
        """Test detection of pie charts."""
        assert detect_diagram_type("pie title Pets\n    Dogs: 50") == "Pie Chart"

    def test_gantt_detection(self):
        """Test detection of gantt charts."""
        assert (
            detect_diagram_type("gantt\n    title Project\n    Task1: 2024-01-01, 30d")
            == "Gantt Chart"
        )

    def test_class_diagram_detection(self):
        """Test detection of class diagrams."""
        assert detect_diagram_type("classDiagram\n    class Animal") == "Class Diagram"

    def test_unknown_diagram_fallback(self):
        """Test fallback for unknown diagram types."""
        assert detect_diagram_type("unknown diagram type\n    content") == "Diagram"
        assert detect_diagram_type("") == "Diagram"
