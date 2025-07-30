from pathlib import Path

# The __file__ variable gives the path to the current file (paths.py)
# .parent gives the directory of the file (src/)
# .parent again gives the project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"