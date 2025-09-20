from typing import Dict, Any


def load_config(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """In a larger code-base this would handle dataset download / graph prep.
    For the refactored sketch-only test we simply pass the parsed YAML config
    through so that other modules receive a uniform Dict[str, Any]."""

    return run_cfg