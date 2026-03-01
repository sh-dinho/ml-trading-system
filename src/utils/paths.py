from pathlib import Path
import os
import logging

def _find_project_root(marker: str = "config") -> Path:
    """
    Dynamically locate the project root by walking up the directory tree
    until a known marker directory/file is found.
    """
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent

    raise RuntimeError("Project root could not be determined.")


# Dynamically resolved root
PROJECT_ROOT = _find_project_root()


# =========================
# Base Directories
# =========================

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
LOGS_DIR = PROJECT_ROOT / "logs"


# =========================
# Path Helpers
# =========================

def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def config_path(*parts: str) -> Path:
    return CONFIG_DIR.joinpath(*parts)


def models_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)


def pipelines_path(*parts: str) -> Path:
    return PIPELINES_DIR.joinpath(*parts)


def logs_path(*parts: str) -> Path:
    return LOGS_DIR.joinpath(*parts)


# =========================
# Utility
# =========================

def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists before writing.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger