from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(base: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = Path(relative)
    return path if path.is_absolute() else base / path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
