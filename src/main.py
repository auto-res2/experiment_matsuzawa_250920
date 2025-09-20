# src/main.py
"""Command-line entry-point orchestrating the project workflow.

The previous iteration only produced a JSON artefact with a *status*
field.  To satisfy the requirement of emitting **concrete experimental
results**, we now capture and serialise the metrics returned by
``evaluate.evaluate``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Project-local imports
from . import evaluate as _ev
from . import preprocess as _pre
from . import train as _tr

# ---------------------------------------------------------------------------
# Constants & paths ----------------------------------------------------------
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"
_RESEARCH_DIR = _REPO_ROOT / ".research" / "iteration2"
_IMAGES_DIR = _RESEARCH_DIR / "images"
_RESULTS_DIR = _RESEARCH_DIR

_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:  # noqa: D401
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_phase(phase_cfg: Dict[str, Any], tag: str) -> None:  # noqa: D401
    """Execute one experiment phase (smoke / full)."""

    result: Dict[str, Any] = {"phase": tag}
    try:
        # 1) Data loading / pre-processing ---------------------------------
        _pre.load_data(phase_cfg)

        # 2) Training -------------------------------------------------------
        _tr.train(phase_cfg)

        # 3) Evaluation -----------------------------------------------------
        metrics = _ev.evaluate(phase_cfg)

        result.update({"status": "success", "metrics": metrics})
    except Exception as err:  # pragma: no cover – any unexpected failure
        result.update(
            {
                "status": "failed",
                "error_type": err.__class__.__name__,
                "message": str(err),
            }
        )
        raise  # honour fail-fast policy
    finally:
        # Persist a JSON artefact for human / CI inspection ---------------
        out_file = _RESULTS_DIR / f"{tag}_result.json"
        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except OSError as io_err:
            sys.stderr.write(f"[WARN] Could not write result file: {io_err}\n")

        # Echo to STDOUT as requested
        print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Main entry -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser("HAWQ-Skim Experiment Runner")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the quick validation scenario.")
    group.add_argument("--full-experiment", action="store_true", help="Run the complete benchmark.")

    args = parser.parse_args()

    # Determine which YAML config to load ----------------------------------
    if args.smoke_test:
        cfg_path = _CONFIG_DIR / "smoke_test.yaml"
        phase_tag = "smoke_test"
    else:  # --full-experiment
        cfg_path = _CONFIG_DIR / "full_experiment.yaml"
        phase_tag = "full_experiment"

    cfg = _load_yaml(cfg_path)

    # Run the selected phase ------------------------------------------------
    _run_phase(cfg, phase_tag)


if __name__ == "__main__":  # pragma: no cover
    main()