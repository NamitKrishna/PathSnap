"""
Smoke tests for PathSnap project structure and module imports.
These tests verify that the project structure is intact and packages can be imported.
"""
import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_project_root_structure():
    """Verify that expected top-level project directories exist."""
    required_dirs = ["data", "notebooks", "src", "tests", "results"]
    for dir_name in required_dirs:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        assert os.path.isdir(dir_path), f"Directory {dir_name} should exist at {dir_path}"


def test_src_subpackages_importable():
    """Verify that src and its core modules can be imported without errors."""
    import src
    import src.preprocessing
    import src.candidate_generation
    import src.models
    import src.optimization
    import src.decoder
    import src.api

    assert src is not None
    assert src.preprocessing is not None
    assert src.candidate_generation is not None
    assert src.models is not None
    assert src.optimization is not None
    assert src.decoder is not None
    assert src.api is not None
