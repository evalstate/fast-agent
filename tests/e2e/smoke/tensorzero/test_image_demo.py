import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("tensorzero_docker_env")


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_tensorzero_image_demo_smoke(project_root, chdir_to_tensorzero_example):
    """
    Smoke test for the TensorZero image demo script.
    Ensures the script runs to completion without errors.
    """
    image_demo_script_path = project_root / "examples" / "tensorzero" / "image_demo.py"

    if not image_demo_script_path.is_file():
        pytest.fail(f"Image demo script not found at {image_demo_script_path}")

    print(f"\nImporting image demo script from: {image_demo_script_path}")
    image_demo_module = None
    try:
        image_demo_module = import_from_path("image_demo_module", image_demo_script_path)
        main_func = getattr(image_demo_module, "main", None)
        if not main_func or not asyncio.iscoroutinefunction(main_func):
            pytest.fail(f"'main' async function not found in {image_demo_script_path}")

        print("Executing image_demo.main()...")
        await main_func()
        print("image_demo.main() executed successfully.")

    except ImportError as e:
        pytest.fail(f"Failed to import image_demo script: {e}")
    except Exception as e:
        pytest.fail(f"Running image_demo script failed: {e}")
    finally:
        if image_demo_module and "image_demo_module" in sys.modules:
            del sys.modules["image_demo_module"]

    print("\nImage demo smoke test completed successfully.")
