ID = "R10"
TAGS = ["spike", "measurement"]
LESSON = ("A candidate scorer's reference must be the query's measured spectrum; a spectrum simulated from the truth "
          "structure ranks the truth first by construction.")
SRC = (".scratch/knapsack_mutator/full_batch_benchmark.py: MetFrag scored mutants against fragment_masses(target_smi); "
       "reported +0.0325 MRR, honest +0.0004 (.scratch/knapsack_mutator/AUDIT.md)")
FLAGS = ["rules/fixtures/R10_truth_reference"]

import ast  # noqa: E402
import re  # noqa: E402

# functions that turn a STRUCTURE into a spectrum / fragment list
SIMULATORS = {"fragment_masses", "frag_scores", "simulate", "predict_spectrum", "iceberg_predict", "cfm_predict",
              "mol_graph"}
TRUTHY = re.compile(r"(truth|target|true|gt)[_a-z]*(smi|smiles|mol)|(smi|smiles)[_a-z]*(truth|target)", re.I)


def _name(f):
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _truthy(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and TRUTHY.search(n.id):
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and TRUTHY.search(n.value):
            return True
        if isinstance(n, ast.Attribute) and TRUTHY.search(n.attr):
            return True
    return False


def check(idea):
    out = []
    for py in sorted((idea / "spike").glob("*.py")) + sorted(idea.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = py.relative_to(idea).as_posix()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and _name(n.func) in SIMULATORS and any(_truthy(a) for a in n.args):
                out.append(f"{rel}:{n.lineno}: {_name(n.func)}() is fed the truth structure — a reference built "
                           f"from the answer; score against the query's measured peaks instead")
    return out
