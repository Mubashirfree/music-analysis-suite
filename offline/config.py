from pathlib import Path

import yaml

PROJECT_ROOT=Path(__file__).resolve().parent.parent
CONFIG_PATH=PROJECT_ROOT / "configs" / "config.yaml"

def load_config(path:Path=CONFIG_PATH)->dict:
    """Read the YAML config file and return it as dictionary."""
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG=load_config()

DATA_RAW =PROJECT_ROOT / CFG["paths"]["data_raw"]
DATA_PROCESSED=PROJECT_ROOT / CFG["paths"]["data_processed"]
DATA_FEATURES=PROJECT_ROOT/ CFG["paths"]["data_features"]
MODELS_DIR=PROJECT_ROOT / CFG["paths"]["models_dir"]
DOCS_DIR=PROJECT_ROOT / CFG["paths"]["docs_dir"]

SEED=CFG["seed"]
SAMPLE_RATE=CFG["audio"]["sample_rate"]
GENRES=CFG["genres"]