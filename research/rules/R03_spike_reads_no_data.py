ID = "R03"
TAGS = ["spike", "measurement"]
LESSON = "A spike must read data (a dump, cached arrays, a log) in its measuring loop; a loop over literal numbers or `pass` is a formula, not a measurement."
SRC = "20261010-fusion-depth-expansion/spike/spike_depth.py: v1 data loop was `pass` (judge), v2 loops over literal rank pairs and never loads its json"
FLAGS = ["exploit/20261010-fusion-depth-expansion", "rules/fixtures/R03_pass_loop"]

import ast  # noqa: E402

READS = {"open", "load", "loads", "read_parquet", "read_csv", "read_json", "read_pickle", "read_text", "load_dump",
         "compare", "fromfile", "genfromtxt", "loadtxt"}


def _name(f):
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def check(idea):
    out = []
    for py in sorted((idea / "spike").glob("*.py")) + sorted(idea.glob("stress_test*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            out.append(f"{py.name}: does not parse ({e.msg})")
            continue
        rel = py.relative_to(idea).as_posix()
        if not any(isinstance(n, ast.Call) and _name(n.func) in READS for n in ast.walk(tree)):
            out.append(f"{rel}: reads no data (no open/load/read_* call)")
        for n in ast.walk(tree):
            if isinstance(n, (ast.For, ast.While)) and any(isinstance(b, ast.Pass) for b in n.body) \
                    and all(isinstance(b, (ast.Pass, ast.Expr)) for b in n.body):
                out.append(f"{rel}:{n.lineno}: loop body is only `pass`")
    return out
