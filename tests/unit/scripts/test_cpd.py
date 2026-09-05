import runpy
import tarfile
from pathlib import Path
from typing import Any

import pytest

_CPD_MODULE = runpy.run_path(str(Path(__file__).parents[3] / "scripts/cpd.py"))
CPDFinding: Any = _CPD_MODULE["CPDFinding"]
cpd_baseline_delta: Any = _CPD_MODULE["cpd_baseline_delta"]
parse_cpd_findings: Any = _CPD_MODULE["parse_cpd_findings"]
ensure_jre: Any = _CPD_MODULE["ensure_jre"]


@pytest.mark.parametrize("system", ["darwin", "linux", "windows"])
@pytest.mark.parametrize("cached", [False, True])
def test_ensure_jre_returns_java_home_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, cached: bool
) -> None:
    tools_dir = tmp_path / "tools"
    bundle_name = f"jdk-{_CPD_MODULE['JRE_VERSION']}-jre"
    home_relative = Path("Contents/Home") if system == "darwin" else Path()
    java_relative = home_relative / "bin" / ("java.exe" if system == "windows" else "java")
    extracted_dir = tools_dir / bundle_name
    source_dir = extracted_dir if cached else tmp_path / "archive-source"
    java_bin = source_dir / java_relative
    java_bin.parent.mkdir(parents=True)
    java_bin.touch()
    tools_dir.mkdir(exist_ok=True)
    if not cached:
        archive = tools_dir / f"{_CPD_MODULE['JRE_FILENAME']}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source_dir, arcname=bundle_name)

    monkeypatch.setitem(ensure_jre.__globals__, "SYSTEM", system)
    monkeypatch.setitem(ensure_jre.__globals__, "TOOLS_DIR", tools_dir)
    monkeypatch.setitem(ensure_jre.__globals__, "JRE_DIR", extracted_dir)
    monkeypatch.setattr("shutil.which", lambda _: None)

    def unexpected_download(*args: Any) -> None:
        pytest.fail("The local JRE or archive should be reused")

    monkeypatch.setitem(ensure_jre.__globals__, "download_file", unexpected_download)
    java_home = ensure_jre()
    assert java_home == extracted_dir / home_relative
    assert (extracted_dir / java_relative).is_file()

    # A subsequent run should use the extracted JRE even without its archive.
    for archive in tools_dir.glob("*.tar.gz"):
        archive.unlink()
    assert ensure_jre() == java_home


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
