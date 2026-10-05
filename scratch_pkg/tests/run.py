"""Run every casmi_scratch test module (no pytest needed): python scratch_pkg/tests/run.py"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
failed = 0
for path in sorted(HERE.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for t in mod.TESTS:
        try:
            t()
        except Exception as e:  # report every failure, then fail the run
            failed += 1
            print(f"FAIL {path.stem}.{t.__name__}: {type(e).__name__}: {e}")
    print(f"{path.stem}: {len(mod.TESTS)} run")
print("scratch_pkg tests: " + ("OK" if not failed else f"{failed} FAILED"))
sys.exit(1 if failed else 0)
