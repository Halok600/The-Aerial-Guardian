"""Test helpers: import numbered scripts (e.g. scripts/01_build_dataset.py) as modules."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
