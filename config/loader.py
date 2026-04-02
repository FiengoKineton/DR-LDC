from pathlib import Path
import yaml

_YAML_PATH = Path(__file__).resolve().parent / "problem___parameters.yaml"


def load_config(path=None):
    yaml_path = Path(path) if path is not None else _YAML_PATH

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
