"""Custom build hook to copy examples to resources during package build."""

import shutil
from pathlib import Path
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to copy examples to resources structure."""

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Copy examples from root to resources structure."""
        # Define source to target mappings
        example_mappings = {
            # examples/workflows -> src/mcp_agent/resources/examples/workflows
            "examples/workflows": "src/mcp_agent/resources/examples/workflows",
            # examples/researcher -> src/mcp_agent/resources/examples/researcher
            "examples/researcher": "src/mcp_agent/resources/examples/researcher",
            # examples/data-analysis -> src/mcp_agent/resources/examples/data-analysis
            "examples/data-analysis": "src/mcp_agent/resources/examples/data-analysis",
            # examples/mcp/state-transfer -> src/mcp_agent/resources/examples/mcp/state-transfer
            "examples/mcp/state-transfer": "src/mcp_agent/resources/examples/mcp/state-transfer",
        }

        print("Fast-agent build: Copying examples to resources...")

        for source_path, target_path in example_mappings.items():
            source_dir = Path(self.root) / source_path
            target_dir = Path(self.root) / target_path

            if source_dir.exists():
                # Ensure target directory exists
                target_dir.parent.mkdir(parents=True, exist_ok=True)

                # Remove existing target if it exists
                if target_dir.exists():
                    shutil.rmtree(target_dir)

                # Copy the entire directory tree
                shutil.copytree(source_dir, target_dir)
                print(f"  Copied {source_path} -> {target_path}")
            else:
                print(f"  Warning: Source directory not found: {source_path}")

        print("Fast-agent build: Example copying completed.")
