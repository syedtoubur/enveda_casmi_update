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
    assert "casmi_scratch/resources/glacier_worker.py" in files
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


def test_generated_notebooks_are_current():
    for driver, name, ovr, stem in mk.NOTEBOOKS:
        dst = PKG / "notebooks" / f"casmi_scratch_{mk.notebook_stem(driver, name, stem)}.ipynb"
        assert dst.exists() and dst.read_text(encoding="utf-8") == mk.notebook_json(driver, name, ovr, stem), \
            f"{dst.name} is stale: python scratch_pkg/tools/mk_notebook.py --all"


def test_r1_twins_and_colab_drivers():
    from py2nb import split_cells, to_nb
    r0 = mk.render("r0_kaggle", tag="r0").split("\n")
    r1b = mk.render("r0_kaggle", {"pool_source": "r1", "pool_drop_e180": True}, tag="r1b").split("\n")
    diff = [(a, b) for a, b in zip(r0, r1b) if a != b]
    assert len(diff) == 2 and "pool_drop_e180=True" in diff[0][1] and "'r1b'" in diff[1][1], diff
    # each GLACIER twin differs from its ICEBERG twin in the override and run-tag lines only
    for ice, gl, tag in (({"pool_source": "r1"}, {"pool_source": "r1", "sim_model": "glacier"}, "r1a"),
                         ({"pool_source": "r1", "pool_drop_e180": True},
                          {"pool_source": "r1", "pool_drop_e180": True, "sim_model": "glacier"}, "r1b")):
        a = mk.render("r0_kaggle", ice, tag=tag).split("\n")
        b = mk.render("r0_kaggle", gl, tag=f"{tag}_glacier").split("\n")
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(a) == len(b) and len(diff) == 2, diff
        assert "sim_model='glacier'" in diff[0][1] and f"'{tag}_glacier'" in diff[1][1], diff
    assert {st for _, _, _, st in mk.NOTEBOOKS} >= {"r1a_glacier", "r1b_glacier"}
    for driver in ("r1_pool_colab", "r1_audit_colab"):
        nb = to_nb(split_cells(mk.render(driver)))
        for i, c in enumerate(["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]):
            ast.parse(c, filename=f"{driver} cell {i}")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for f in TESTS:
        f()
    print(f"test_notebook: {len(TESTS)} OK")
