"""
candidate_generation package initialization.
Exports all functions and classes from the candidate_generation module
to support both module and package import styles seamlessly.
"""

from pathlib import Path
import importlib.util

_module_path = Path(__file__).resolve().parent.parent / "candidate_generation.py"
if _module_path.exists():
    _spec = importlib.util.spec_from_file_location("src._candidate_generation_mod", _module_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _attr in dir(_mod):
        if not _attr.startswith("_"):
            globals()[_attr] = getattr(_mod, _attr)
