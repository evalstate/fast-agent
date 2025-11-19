"""Custom build hook to copy examples to resources during package build."""

import shutil
from pathlib import Path
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to copy examples to resources structure."""

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Copy examples from root to resources structure."""
        # Clear existing resources/examples directory for clean build
        resources_examples_dir = Path(self.root) / "src/fast_agent/resources/examples"
        if resources_examples_dir.exists():
            shutil.rmtree(resources_examples_dir)
            print("Fast-agent build: Cleared existing resources/examples directory")

        # Clear existing fast_agent setup resources for clean build
        resources_setup_dir = Path(self.root) / "src/fast_agent/resources/setup"
        if resources_setup_dir.exists():
            shutil.rmtree(resources_setup_dir)
            print("Fast-agent build: Cleared existing resources/setup directory")

        # Define source to target mappings
        # Examples:
        #  examples/workflows -> src/fast_agent/resources/examples/workflows
        #  examples/mcp/elicitations -> src/fast_agent/resources/examples/mcp/elicitations
        EXAMPLES_DIR = Path("src") / "fast_agent" / "resources" / "examples"
        example_mappings = {
            "examples/workflows": EXAMPLES_DIR / "workflows",
            "examples/researcher": EXAMPLES_DIR / "researcher",
            "examples/data-analysis": EXAMPLES_DIR / "data-analysis",
            "examples/mcp/state-transfer": EXAMPLES_DIR / "mcp" / "state-transfer",
            "examples/mcp/elicitations": EXAMPLES_DIR / "mcp" / "elicitations",
            "examples/tensorzero": EXAMPLES_DIR / "tensorzero",
        }

        # Define setup template mapping (editable templates -> packaged resources)
        setup_mappings = {
            # examples/setup -> src/fast_agent/resources/setup
            "examples/setup": "src/fast_agent/resources/setup",
        }

        print("Fast-agent build: Copying examples to resources...")

        for source_path, target_path in example_mappings.items():
            source_dir = Path(self.root) / source_path
            target_dir = Path(self.root) / target_path

            if source_dir.exists():
                # Ensure target directory exists
                target_dir.parent.mkdir(parents=True, exist_ok=True)

                # Copy the entire directory tree
                shutil.copytree(source_dir, target_dir)
                print(f"  Copied {source_path} -> {target_path}")
            else:
                print(f"  Warning: Source directory not found: {source_path}")

        print("Fast-agent build: Example copying completed.")

        # Copy setup templates
        print("fast-agent build: Copying setup templates to resources...")
        for source_path, target_path in setup_mappings.items():
            source_dir = Path(self.root) / source_path
            target_dir = Path(self.root) / target_path

            if source_dir.exists():
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_dir, target_dir)
                print(f"  Copied {source_path} -> {target_path}")
            else:
                print(f"  Warning: Setup templates directory not found: {source_path}")

        # Ensure the copied files are included in the build
        if "artifacts" not in build_data:
            build_data["artifacts"] = []

        # Add all copied files as artifacts
        for target_path in list(example_mappings.values()) + list(setup_mappings.values()):
            target_dir = Path(self.root) / target_path
            if target_dir.exists():
                # Add all files in the target directory recursively
                for file_path in target_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = str(file_path.relative_to(Path(self.root)))
                        if relative_path not in build_data["artifacts"]:
                            build_data["artifacts"].append(relative_path)
                            print(f"  Added artifact: {relative_path}")
