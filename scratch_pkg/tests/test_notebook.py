"""The generated notebook embeds the current package, parses, and a probe differs from its control in one line."""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "tools"))
import mk_notebook as mk  # noqa: E402


def test_render_and_embed():
    text = mk.render("r0_kaggle")
    tree = ast.parse(text)
    files = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PKG_FILES")
    assert files == mk.package_files()
    assert "casmi_scratch/resources/iceberg_worker.py" in files and "casmi_scratch/pipeline.py" in files
    # the embedded package imports cleanly on its own (written the way the notebook writes it)
    with tempfile.TemporaryDirectory() as d:
        for rel, src in files.items():
            p = Path(d) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        r = subprocess.run([sys.executable, "-c", "import casmi_scratch.pipeline, casmi_scratch.sim; print('ok')"],
                           cwd=d, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_cells_parse_and_override():
    from py2nb import split_cells, to_nb
    base = mk.render("r0_kaggle", tag="r0")
    probe = mk.render("r0_kaggle", {"use_sim": False}, tag="r0")
    diff = [(a, b) for a, b in zip(base.split("\n"), probe.split("\n")) if a != b]
    assert len(diff) == 1 and "use_sim=False" in diff[0][1], diff
    nb = to_nb(split_cells(base))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code):
        ast.parse(c, filename=f"cell {i}")
    json.dumps(nb)
    assert mk.parse_set(["use_sim=False", "tau=0.8"]) == {"use_sim": False, "tau": 0.8}


def test_generated_notebook_is_current():
    from py2nb import split_cells, to_nb
    dst = PKG / "notebooks" / "casmi_scratch_r0.ipynb"
    want = json.dumps(to_nb(split_cells(mk.render("r0_kaggle", tag="r0"))), ensure_ascii=False, indent=1)
    assert dst.exists() and dst.read_text(encoding="utf-8") == want, \
        "casmi_scratch_r0.ipynb is stale: python scratch_pkg/tools/mk_notebook.py r0_kaggle"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for f in TESTS:
        f()
    print(f"test_notebook: {len(TESTS)} OK")
