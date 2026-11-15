import os
from pathlib import Path

# Base repository root (inferred dynamically from this file location)
# Works identically across Windows, macOS, Linux, Colab, Kaggle
ROOT_DIR = Path(__file__).resolve().parent.parent

# Core subdirectories (relative to ROOT_DIR)
CONFIGS_DIR = ROOT_DIR / "configs"
SCHEMAS_DIR = ROOT_DIR / "schemas"
SRC_DIR = ROOT_DIR / "src"
TESTS_DIR = ROOT_DIR / "tests"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"

# Standard data stages (all gitignored)
SEEDS_RAW_DIR = DATA_DIR / "01_seeds_raw"
EXPANDED_DIR = DATA_DIR / "02_expanded"
CURATED_DIR = DATA_DIR / "03_curated"
EVAL_CACHE_DIR = DATA_DIR / "04_eval_cache"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"

# Optional external locations (configurable via environment variables)
# If STRATA or manual traces live in a sibling directory or shared drive:
STRATA_DIR = Path(os.getenv("STRATA_DIR", ROOT_DIR.parent / "STRATA"))
MANUAL_TRACES_DIR = Path(os.getenv("MANUAL_TRACES_DIR", SEEDS_RAW_DIR))

def ensure_dirs():
    """Ensure all required local directories exist."""
    for d in [CACHE_DIR, DATA_DIR, SEEDS_RAW_DIR, EXPANDED_DIR, CURATED_DIR, EVAL_CACHE_DIR, BENCHMARKS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

