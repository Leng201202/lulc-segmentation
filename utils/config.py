from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(base: Path, relative: str | Path | None) -> Path | None:
    """Resolve a possibly-relative path against a base directory.

    - If `relative` is None/empty, returns None.
    - If `relative` is already an absolute path, returns it as-is.
    - Otherwise, returns `base / relative`.
    """
    if relative is None or relative == "":
        return None
    path = Path(relative)
    if path.is_absolute():
        return path
    return base / path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
