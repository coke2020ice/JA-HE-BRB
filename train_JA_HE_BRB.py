from pathlib import Path
import runpy, sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    runpy.run_path(str(ROOT / "examples/run_reference.py"), run_name="__main__")
