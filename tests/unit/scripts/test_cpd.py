import runpy
from pathlib import Path
from typing import Any

import pytest

_CPD_MODULE = runpy.run_path(str(Path(__file__).parents[3] / "scripts/cpd.py"))
CPDFinding: Any = _CPD_MODULE["CPDFinding"]
cpd_baseline_delta: Any = _CPD_MODULE["cpd_baseline_delta"]
parse_cpd_findings: Any = _CPD_MODULE["parse_cpd_findings"]


def test_parse_cpd_findings_ignores_line_numbers_and_normalizes_paths(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    first = src_dir / "package" / "first.py"
    second = src_dir / "package" / "second.py"
    xml = f"""\
<pmd-cpd xmlns="https://pmd-code.org/schema/cpd-report">
  <duplication lines="3" tokens="120">
    <file path="{first}" line="10" endline="12"/>
    <file path="{second}" line="40" endline="42"/>
    <codefragment>same\r
code</codefragment>
  </duplication>
</pmd-cpd>
"""

    findings = parse_cpd_findings(xml, src_dir)

    assert len(findings) == 1
    finding = next(iter(findings))
    assert finding.tokens == 120
    assert finding.paths == ("package/first.py", "package/second.py")
    assert len(finding.code_hash) == 64


def test_cpd_baseline_delta_reports_new_and_stale_findings() -> None:
    approved = CPDFinding(tokens=120, paths=("a.py", "b.py"), code_hash="approved")
    stale = CPDFinding(tokens=110, paths=("c.py", "d.py"), code_hash="stale")
    new = CPDFinding(tokens=130, paths=("e.py", "f.py"), code_hash="new")

    unapproved, stale_findings = cpd_baseline_delta(
        frozenset({approved, new}),
        frozenset({approved, stale}),
        min_tokens=100,
    )

    assert unapproved == frozenset({new})
    assert stale_findings == frozenset({stale})


@pytest.mark.parametrize(
    "xml",
    [
        '<error xmlns="https://pmd-code.org/schema/cpd-report"/>',
        '<pmd-cpd xmlns="urn:unexpected-schema"/>',
        '<pmd-cpd><duplication tokens="120"/></pmd-cpd>',
    ],
)
def test_parse_cpd_findings_rejects_unexpected_report_root(xml: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Expected CPD report root"):
        parse_cpd_findings(xml, tmp_path)


def test_parse_cpd_findings_accepts_empty_report(tmp_path: Path) -> None:
    xml = '<pmd-cpd xmlns="https://pmd-code.org/schema/cpd-report"/>'

    assert parse_cpd_findings(xml, tmp_path) == frozenset()
