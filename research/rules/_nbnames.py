"""Static check: module-level names a notebook cell reads must be bound in an earlier cell or in that cell.

Function and lambda bodies are skipped (late binding is legal there). IPython lines (`!`, `%`) are dropped.
Used by rule R07 and by the exploit worktree's build/test_patch_harness_dump.py.
"""
from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"__file__", "get_ipython", "display", "In", "Out", "exit", "quit"}


def _parse(src: str):
    keep = [ln if not ln.lstrip().startswith(("!", "%")) else "" for ln in src.splitlines()]
    try:
        return ast.parse("\n".join(keep))
    except SyntaxError:
        return None


class _Walk(ast.NodeVisitor):
    def __init__(self):
        self.bound, self.loads = set(), []

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load):
            self.loads.append((n.id, n.lineno))
        else:
            self.bound.add(n.id)

    def visit_Import(self, n):
        for a in n.names:
            self.bound.add(a.asname or a.name.split(".")[0])

    visit_ImportFrom = visit_Import

    def _def(self, n):
        self.bound.add(n.name)
        for d in getattr(n, "decorator_list", []):
            self.visit(d)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def

    def visit_Lambda(self, n):
        pass

    def visit_Global(self, n):
        self.bound.update(n.names)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_comprehension(self, n):
        self.visit(n.target)
        self.visit(n.iter)
        for i in n.ifs:
            self.visit(i)


def _bind_all(tree) -> set[str]:
    """Every name a cell binds anywhere, incl. `global` writes inside functions."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def unbound(cells: list[str]) -> list[tuple[int, str, int]]:
    """[(cell index among the given code cells, name, line)] read before any binding."""
    seen, out = set(BUILTINS), []
    for i, src in enumerate(cells):
        tree = _parse(src)
        if tree is None:
            continue
        w = _Walk()
        for stmt in tree.body:
            w.visit(stmt)
        cell_bound = _bind_all(tree)
        for name, ln in w.loads:
            if name not in seen and name not in cell_bound:
                out.append((i, name, ln))
        seen |= cell_bound
        if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
            seen.add("*")
    return [] if "*" in seen else out


def code_cells(ipynb: Path) -> list[str]:
    nb = json.loads(Path(ipynb).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


# ------------------------------------------------------------------------------------------------
# Harness helper names re-bound by a base cell (harness-c1npx v1: SNAP's `pickle as _pk`, then the base's merge
# self-test `_b, _bk, _p, _pk = ...` -> `'list' object has no attribute 'dump'` in the final cell)
# ------------------------------------------------------------------------------------------------
HX_CELL = "# === HARNESS"                       # a whole inserted cell starts with this
HX_BEGIN, HX_END = "# >>> HARNESS", "# <<< HARNESS"   # an inline snippet inside a base cell


def harness_blocks(cells: list[str]) -> list[tuple[int, str]]:
    """[(cell index, harness source)]: whole `# === HARNESS` cells and `# >>> HARNESS` ... `# <<< HARNESS` snippets."""
    out = []
    for i, src in enumerate(cells):
        if src.lstrip().startswith(HX_CELL):
            out.append((i, src))
            continue
        lines, cur = src.splitlines(keepends=True), None
        for ln in lines:
            if ln.lstrip().startswith(HX_BEGIN):
                cur = [ln]
            elif cur is not None:
                cur.append(ln)
                if ln.lstrip().startswith(HX_END):
                    out.append((i, "".join(cur)))
                    cur = None
    return out


def _module_binds(tree) -> set[str]:
    """Names bound at module level (incl. inside if/for/try/with), not inside function or class bodies."""
    out = set()

    def walk(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(ch.name)
                continue
            if isinstance(ch, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                continue
            if isinstance(ch, ast.Name) and not isinstance(ch.ctx, ast.Load):
                out.add(ch.id)
            elif isinstance(ch, (ast.Import, ast.ImportFrom)):
                out.update(a.asname or a.name.split(".")[0] for a in ch.names)
            elif isinstance(ch, ast.ExceptHandler) and ch.name:
                out.add(ch.name)
            walk(ch)
    walk(tree)
    return out


def _func_free_loads(fn) -> set[str]:
    """Names a function reads that are neither its parameters nor names it binds itself (= globals at call time)."""
    a = fn.args
    local = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    local |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    loads = set()
    for m in ast.walk(fn):
        if m is fn:
            continue
        if isinstance(m, ast.Name):
            (loads if isinstance(m.ctx, ast.Load) else local).add(m.id)
        elif isinstance(m, (ast.Import, ast.ImportFrom)):
            local.update(x.asname or x.name.split(".")[0] for x in m.names)
        elif isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(m.name)
        elif isinstance(m, ast.arg):
            local.add(m.arg)
    return loads - local


def _func_loads(tree, called) -> set[str]:
    """Global reads of the block's module-level functions that later harness code calls (late binding)."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in called:
            out |= _func_free_loads(n)
    return out


def _all_loads(tree) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _imports(tree) -> dict[str, set[str]]:
    """name -> modules it is bound to by import statements anywhere in the tree."""
    out: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.setdefault(a.asname or a.name.split(".")[0], set()).add(a.name if a.asname else a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.setdefault(a.asname or a.name, set()).add(f"{n.module}.{a.name}")
    return out


def _non_import_binds(tree) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def rebound(cells: list[str], blocks=None) -> list[tuple[int, str, int]]:
    """[(harness cell, name, base cell)]: a name a harness block binds and harness code reads later (inside its
    functions, or in a later harness block) that a base cell at or after the block re-binds."""
    blocks = harness_blocks(cells) if blocks is None else blocks
    trees = [(i, _parse(b)) for i, b in blocks]
    base = []
    for c, src in enumerate(cells):
        rest = src
        for i, b in blocks:
            if i == c:
                rest = rest.replace(b, "")
        t = _parse(rest) if rest.strip() else None
        base.append((_non_import_binds(t), _imports(t)) if t is not None else (set(), {}))
    out = []
    for k, (i, t) in enumerate(trees):
        if t is None:
            continue
        later = set()
        for _, t2 in trees[k + 1:]:
            if t2 is not None:
                later |= _all_loads(t2)
        danger = _module_binds(t) & (_func_loads(t, later) | later)
        mine = _imports(t)
        for c in range(i, len(cells)):
            plain, imps = base[c]
            for name in sorted(danger):
                # re-importing the very same module under the same name (`import pandas as pd`) is harmless
                if name in plain or (name in imps and imps[name] != mine.get(name)):
                    out.append((i, name, c))
    return sorted(set(out))
