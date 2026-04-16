from pathlib import Path
import yaml, copy, pprint, sys
from .interactive import apply_terminal_overrides, validate_cfg


_CONFIG_PATH = Path(__file__).resolve().parent / "problem___parameters"


def _deep_update(base: dict, new: dict) -> dict:
    """
    Recursively merge dictionary `new` into dictionary `base`.
    If a key exists in both:
      - if both values are dicts, merge recursively
      - otherwise, overwrite base with new
    """
    for key, value in new.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a dict: {path}")
    return data


def load_config(path=None):
    """
    Load configuration from:
      - a single YAML file, or
      - a directory containing multiple YAML files

    If a directory is provided, all *.yaml files are loaded in sorted order
    and merged under a single top-level key: 'params'.

    Split YAML files must NOT contain 'params:' themselves.
    """
    config_path = Path(path) if path is not None else _CONFIG_PATH

    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if config_path.is_dir():
        merged_params = {}

        yaml_files = sorted(config_path.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in directory: {config_path}")

        file_to_section = {
            "00_core.yaml": None,
            "01_flags.yaml": None,
            "02_directories.yaml": "directories",
            "03_ambiguity.yaml": "ambiguity",
            "04_simulation.yaml": "simulation",
            "05_ident.yaml": "ident",
            "06_plant.yaml": "plant",
            "07_outputs.yaml": "outputs",
        }

        for yaml_file in yaml_files:
            part = _load_yaml_file(yaml_file)
            section = file_to_section.get(yaml_file.name)

            if section is None:
                _deep_update(merged_params, part)
            else:
                if section not in merged_params:
                    merged_params[section] = {}
                _deep_update(merged_params[section], part)

        return {"params": merged_params}

    raise FileNotFoundError(f"Config path not found: {config_path}")




def build_runtime_config():
    cfg_init = load_config()

    while True:
        cfg = apply_terminal_overrides(copy.deepcopy(cfg_init))
        try:
            validate_cfg(cfg)
            return cfg
        except ValueError as e:
            print(f"\nInvalid configuration: {e}")
            print("Please try again.\n")
