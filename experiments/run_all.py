import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carex.analysis import run_all
print(run_all(Path(__file__).resolve().parents[1]))
