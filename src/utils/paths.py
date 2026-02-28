from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def data_path(*args):
    """Construct a path to the data directory."""
    return PROJECT_ROOT / 'data' / Path(*args)

def config_path(*args):
    """Construct a path to the config directory."""
    return PROJECT_ROOT / 'config' / Path(*args)

def models_path(*args):
    """Construct a path to the models directory."""
    return PROJECT_ROOT / 'models' / Path(*args)

def pipelines_path(*args):
    """Construct a path to the pipelines directory."""
    return PROJECT_ROOT / 'pipelines' / Path(*args)

