import json
from pathlib import Path


def compute_ra(corrupted_map: float, clean_map: float) -> float:
    """Resistance Ability = corrupted mAP / clean mAP (MultiCorrupt definition)."""
    if clean_map == 0:
        raise ValueError("clean_map is zero — baseline must be validated before computing RA")
    return corrupted_map / clean_map


def load_map_results(results_dir: Path) -> dict:
    """Load all saved mAP JSON files from a results directory into a nested dict."""
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        results[f.stem] = data
    return results


def build_ra_table(corruption_results: dict, clean_map: float) -> dict:
    """Given a dict of {condition_key: map_value}, return {condition_key: ra_value}."""
    return {k: compute_ra(v, clean_map) for k, v in corruption_results.items()}
