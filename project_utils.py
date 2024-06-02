from pathlib import Path


def project_dir() -> Path:
    """Returns the path of the root directory."""
    return Path(__file__).resolve().parent


def datasets_dir() -> Path:
    """Returns the path of the `datasets` directory."""
    return Path(__file__).resolve().parent / "datasets"
