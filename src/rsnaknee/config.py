from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    root = path.parent.parent
    for key in ("raw_dir", "artifacts_dir"):
        config[key] = root / config[key]
    return config
