"""No real dataset is shipped with the public snippet.  This module therefore
provides stub helpers that would ordinarily download / transform Reddit, OGBN
etc.  For CI-smoke purposes we just return an empty dict but keep the call-site
intact so future extensions can plug real data-loading here without touching
main.py."""

__all__ = ["load_data"]

def load_data(cfg):
    if cfg.get("verbose", True):
        print("preprocess.load_data – no-op (dataset handled in full code-base)")
    return {}
