#!/usr/bin/env python3
"""
Copy/Paste Detector (CPD) runner for fast-agent.

Uses PMD's CPD tool to detect duplicated code in the Python source.
Automatically downloads Java JRE and PMD if not present.

Usage:
    uv run scripts/cpd.py [--min-tokens N] [--format FORMAT] [--report FILE]

Options:
    --min-tokens N   Minimum token count for duplication (default: 100)
    --format FORMAT  Output format: text, csv, xml (default: text)
    --report FILE    Write report to file (default: stdout)
    --check          Exit with error code if duplications found (for CI)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Tool versions and URLs
JRE_VERSION = "17.0.9+9"
PMD_VERSION = "7.9.0"

TOOLS_DIR = Path.home() / "tools"
JRE_DIR = TOOLS_DIR / f"jdk-{JRE_VERSION}-jre"
PMD_DIR = TOOLS_DIR / f"pmd-bin-{PMD_VERSION}"

# Platform-specific JRE download
SYSTEM = platform.system().lower()
ARCH = platform.machine().lower()

if ARCH in ("x86_64", "amd64"):
    ARCH_LABEL = "x64"
elif ARCH in ("aarch64", "arm64"):
    ARCH_LABEL = "aarch64"
else:
    ARCH_LABEL = ARCH

if SYSTEM == "darwin":
    OS_LABEL = "mac"
elif SYSTEM == "linux":
    OS_LABEL = "linux"
elif SYSTEM == "windows":
    OS_LABEL = "windows"
else:
    OS_LABEL = SYSTEM

JRE_FILENAME = f"OpenJDK17U-jre_{ARCH_LABEL}_{OS_LABEL}_hotspot_{JRE_VERSION.replace('+', '_')}"
JRE_URL = f"https://github.com/adoptium/temurin17-binaries/releases/download/jdk-{JRE_VERSION.replace('+', '%2B')}/{JRE_FILENAME}.tar.gz"
PMD_URL = f"https://github.com/pmd/pmd/releases/download/pmd_releases%2F{PMD_VERSION}/pmd-dist-{PMD_VERSION}-bin.zip"


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress indication."""
    print(f"Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Downloaded to {dest}")
    except Exception as e:
        print(f"  Failed to download: {e}", file=sys.stderr)
        sys.exit(1)


def ensure_jre() -> Path:
    """Ensure Java JRE is available, downloading if necessary."""
    java_bin = JRE_DIR / "bin" / "java"
    if SYSTEM == "windows":
        java_bin = java_bin.with_suffix(".exe")

    if java_bin.exists():
        return JRE_DIR

    # Check system Java
    system_java = shutil.which("java")
    if system_java:
        try:
            result = subprocess.run(
                [system_java, "-version"],
                capture_output=True,
                text=True,
            )
            version_output = result.stderr + result.stdout
            if "17" in version_output or "21" in version_output:
                print(f"Using system Java: {system_java}")
                return Path(system_java).parent.parent
        except Exception:
            pass

    # Download JRE
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / f"{JRE_FILENAME}.tar.gz"

    if not archive_path.exists():
        download_file(JRE_URL, archive_path, f"Java JRE {JRE_VERSION}")

    print("Extracting Java JRE...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(TOOLS_DIR)

    # Handle different directory naming conventions
    extracted_dir = TOOLS_DIR / f"jdk-{JRE_VERSION}-jre"
    if not extracted_dir.exists():
        # Try alternate naming
        for d in TOOLS_DIR.iterdir():
            if d.is_dir() and d.name.startswith("jdk-17"):
                extracted_dir = d
                break

    return extracted_dir


def ensure_pmd() -> Path:
    """Ensure PMD is available, downloading if necessary."""
    pmd_bin = PMD_DIR / "bin" / "pmd"
    if SYSTEM == "windows":
        pmd_bin = PMD_DIR / "bin" / "pmd.bat"

    if pmd_bin.exists():
        return PMD_DIR

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TOOLS_DIR / f"pmd-{PMD_VERSION}.zip"

    if not archive_path.exists():
        download_file(PMD_URL, archive_path, f"PMD {PMD_VERSION}")

    print("Extracting PMD...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(TOOLS_DIR)

    # Make pmd executable on Unix
    if SYSTEM != "windows":
        pmd_bin.chmod(0o755)

    return PMD_DIR


def run_cpd(
    java_home: Path,
    pmd_dir: Path,
    src_dir: Path,
    min_tokens: int = 100,
    output_format: str = "text",
) -> tuple[int, str]:
    """Run CPD and return exit code and output."""
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = f"{java_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    pmd_bin = pmd_dir / "bin" / "pmd"
    if SYSTEM == "windows":
        pmd_bin = pmd_dir / "bin" / "pmd.bat"

    cmd = [
        str(pmd_bin),
        "cpd",
        "--language", "python",
        "--minimum-tokens", str(min_tokens),
        "--dir", str(src_dir),
        "--format", output_format,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr
    return result.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect duplicated code in fast-agent source"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="Minimum token count for duplication (default: 100)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "xml"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with error code if duplications found (for CI)",
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / "src"

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    # Ensure tools are available
    print("Checking dependencies...")
    java_home = ensure_jre()
    pmd_dir = ensure_pmd()
    print()

    # Run CPD
    print(f"Running CPD on {src_dir} (min-tokens={args.min_tokens})...")
    print()

    exit_code, output = run_cpd(
        java_home=java_home,
        pmd_dir=pmd_dir,
        src_dir=src_dir,
        min_tokens=args.min_tokens,
        output_format=args.format,
    )

    if args.report:
        args.report.write_text(output)
        print(f"Report written to {args.report}")
    else:
        print(output)

    # CPD exit codes: 0 = no duplication, 4 = duplications found
    if exit_code == 4:
        print("\n⚠️  Duplicated code detected!")
        if args.check:
            return 1
        return 0
    elif exit_code == 0:
        print("\n✅ No duplicated code found.")
        return 0
    else:
        print(f"\n❌ CPD failed with exit code {exit_code}", file=sys.stderr)
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
