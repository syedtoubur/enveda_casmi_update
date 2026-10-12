"""Replay harness: measure a post-scoring change of the 0.433 base on val-mode dumps, on CPU, in seconds.

    python research/harness/harness.py fetch    <instrument idea>             # kernel output -> dumps/<shard>/
    python research/harness/harness.py compare  --dump C2 --b "CELL19_ICE=True" [--a ""] [--post build/casmi_post.py]
    python research/harness/harness.py lossmap  --dump C2 [--dump C1_npx]      # -> research/harness/loss_map.md
    python research/harness/harness.py backtest                                # H2 flips -> gate_report.md Results
    python research/harness/harness.py warm     --dump C2 [--procs 7]          # fill dumps/<shard>/ident_cache.pkl once
    python research/harness/harness.py suite    --dump C2 --dump C1_npx [--eng-off]  # back-tests -> capability_map.md/.json

A dump is post_dump/ from a build/patch_harness_dump.py run: state.pkl plus the fuse.py / gl_fuse.py / casmi/*.py
sources the replay needs. --post defaults to the exploit worktree's build/casmi_post.py, so a probe that adds a knob
there is measured with exactly the code that would ship. Truth comes from val_scripts/val_key.pkl.
Read research/harness/gate_report.md first: deltas only (L07), post-scoring scope, leak proxy.
`--json out.json` on compare writes the machine-readable result (harness_result.json for an idea folder).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DUMPS = HERE / "dumps"
VAL_KEY = REPO / "val_scripts" / "val_key.pkl"
WORKTREE = Path("C:/casmi/lines/exploit")
SHARD_OF = {"20261010-harness-c2": "C2", "20261010-harness-c1npx": "C1_npx", "20261010-harness-draft": "draft"}
BACKTEST = [("CELL19_ICE=True", -1), ("USE_ENG=False", -1), ("FUSED_GL=False", -1), ("PROMOTE=False", -1)]


# ---------------------------------------------------------------- loading
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- identity cache
# chem.score_key and casmi_post.canonicalize_smiles_to_ik14 (tautomer canonicalisation, ~55 ms each) are >99% of a
# replay. Both are pure in the SMILES, so their results persist per dump in dumps/<shard>/ident_cache.pkl, tagged with
# the RDKit version. `harness.py warm --dump X` fills it on all cores; every command saves what it added.
_CACHES: dict[str, dict] = {}
_FNS = ("score_key", "ik14")


def _cache(shard: str) -> dict:
    if shard not in _CACHES:
        from rdkit import rdBase
        p = DUMPS / shard / "ident_cache.pkl"
        c = pickle.load(open(p, "rb")) if p.exists() else {}
        if c.get("rdkit") != rdBase.rdkitVersion:
            c = {"rdkit": rdBase.rdkitVersion, **{f: {} for f in _FNS}}
        _CACHES[shard] = dict(path=p, data=c, n0=sum(len(c[f]) for f in _FNS))
    return _CACHES[shard]


def _cached(fn, table: dict):
    def wrapped(s):
        try:
            return table[s]
        except (KeyError, TypeError):
            v = fn(s)
            if isinstance(s, str):
                table[s] = v
            return v
    return wrapped


def save_caches() -> None:
    for shard, c in _CACHES.items():
        n = sum(len(c["data"][f]) for f in _FNS)
        if n != c["n0"]:
            tmp = c["path"].with_suffix(".tmp")
            pickle.dump(c["data"], open(tmp, "wb"), protocol=4)
            tmp.replace(c["path"])
            c["n0"] = n
            print(f"ident cache {shard}: {n} entries saved", file=sys.stderr)


def load_dump(shard: str, post: Path | None = None):
    d = DUMPS / shard / "post_dump"
    state = pickle.load(open(d / "state.pkl", "rb"))
    src = str(d / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import fuse, gl_fuse  # noqa: E401  (the dump's own copies)
    from casmi import chem
    cp = load_module(post or (WORKTREE / "build" / "casmi_post.py"), "casmi_post_under_test")
    cache = _cache(shard)["data"]
    cp.canonicalize_smiles_to_ik14 = _cached(cp.canonicalize_smiles_to_ik14, cache["ik14"])
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    def form(s):
        try:
            return CalcMolFormula(Chem.MolFromSmiles(s))
        except Exception:
            return s
    deps = dict(score_key=_cached(chem.score_key, cache["score_key"]), form=form, ice_rerank=fuse.rerank,
                gl_rerank_multi=gl_fuse.rerank_multi, dreams=None)
    return state, deps, cp


def _state_smiles(state) -> list[str]:
    out = set()
    for v in state["base"].values():
        out.update(v[0])
    for p in (state.get("pc") or {}).values():
        out.update((p or {}).get("pc") or [])
    for e in (state.get("eng") or {}).values():
        out.update((e or {}).get("smiles") or [])
    return sorted(s for s in out if isinstance(s, str))


def _warm_init(src: str, post: str):
    global _W
    sys.path.insert(0, src)
    from casmi import chem
    _W = (chem.score_key, load_module(Path(post), "casmi_post_worker").canonicalize_smiles_to_ik14)


def _warm_one(s: str):
    return s, _W[0](s), _W[1](s)


def cmd_warm(shard: str, procs: int, post: Path | None = None) -> int:
    from multiprocessing import Pool
    state, _, _ = load_dump(shard, post)
    T = truths()
    cache = _cache(shard)["data"]
    todo = [s for s in _state_smiles(state) + [T[m]["smiles"] for m in state["te_mids"] if m in T]
            if s not in cache["score_key"] or s not in cache["ik14"]]
    print(f"warm {shard}: {len(todo)} SMILES on {procs} processes", flush=True)
    src = str(DUMPS / shard / "post_dump" / "src")
    with Pool(procs, _warm_init, (src, str(post or (WORKTREE / "build" / "casmi_post.py")))) as pool:
        for i, (s, k, ik) in enumerate(pool.imap_unordered(_warm_one, todo, chunksize=64), 1):
            cache["score_key"][s], cache["ik14"][s] = k, ik
            if i % 5000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                save_caches()
    save_caches()
    return 0


def truths() -> dict[str, dict]:
    return {r["molecule_id"]: r for r in pickle.load(open(VAL_KEY, "rb"))}


def parse_set(spec: str) -> dict:
    out = {}
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        out[k.strip()] = json.loads(v.lower() if v.lower() in ("true", "false") else v)
    return out


# ---------------------------------------------------------------- metrics
def rank_of(cands: list[str], truth_ik: str, cp) -> int:
    for i, s in enumerate(cands[:25], 1):
        if cp.canonicalize_smiles_to_ik14(s) == truth_ik:
            return i
    return 0


def rr(ranks: np.ndarray) -> np.ndarray:
    return np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)


def bootstrap(delta: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not len(delta):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def fp_oracle(state, mid: str, truth_ik: str, cp) -> bool:
    """Leak proxy: an FPNet-driven list (PubChem f.z list, Engine 2) already has the truth at rank 1."""
    p = (state.get("pc") or {}).get(mid) or {}
    e = (state.get("eng") or {}).get(mid) or {}
    tops = [x[0] for x in (p.get("pc") or [], e.get("smiles") or []) if x]
    return any(cp.canonicalize_smiles_to_ik14(s) == truth_ik for s in tops)


def ranks_for(state, deps, cp, overrides: dict, mids: list[str], tk: dict) -> np.ndarray:
    out = cp.run(state, deps, **overrides)
    return np.array([rank_of(out.get(m, []), tk[m], cp) for m in mids])


def compare(shard: str, a: dict, b: dict, post: Path | None = None) -> dict:
    state, deps, cp = load_dump(shard, post)
    T = truths()
    mids = [m for m in state["te_mids"] if m in T]
    tk = {m: cp.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in mids}
    ra, rb = ranks_for(state, deps, cp, a, mids, tk), ranks_for(state, deps, cp, b, mids, tk)
    d = rr(rb) - rr(ra)
    orc = np.array([fp_oracle(state, m, tk[m], cp) for m in mids])
    lo, hi = bootstrap(d)
    res = dict(shard=shard, n=len(mids), a=a, b=b, mrr_a=float(rr(ra).mean()), mrr_b=float(rr(rb).mean()),
               delta=float(d.mean()), ci90=[lo, hi], up=int((d > 0).sum()), down=int((d < 0).sum()),
               delta_no_oracle=float(d[~orc].mean()) if (~orc).any() else float("nan"), n_oracle=int(orc.sum()))
    res["leak_sensitive"] = bool(np.sign(res["delta"]) != np.sign(res["delta_no_oracle"]) and res["delta"] != 0)
    res["family"], res["verdict"] = capmap_verdict(b)
    res["lb_slot"] = bool(res["delta"] >= 0.005 and lo > 0 and not res["leak_sensitive"]
                          and res["verdict"] not in NOT_CITABLE)
    return res


# ---------------------------------------------------------------- back-test suite -> capability map
# Rules pre-registered in gate_report.md (2026-10-11, "back-test suite + capability map"); spec in backtests.yaml.
SPEC = HERE / "backtests.yaml"
CAPMAP = HERE / "capability_map.json"
EVIDENCE, KEYED = ("S", "A"), ("S", "A", "B")
ORDER = ["BLIND", "BASE-COUPLED", "ADVISORY", "TRUSTED-PROVISIONAL", "TRUSTED"]
NOT_CITABLE = ("BLIND", "BASE-COUPLED")
COUPLING_TH = 0.05     # gate_report 2026-10-12 "Option 2 BASE-coupling rule" (pre-registered)


def load_spec(path: Path = SPEC) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def knob_families(spec: dict) -> dict[str, str]:
    return {k: fam for fam, knobs in spec["families"].items() for k in knobs}


def family_of(knobs: dict, spec: dict) -> str:
    """An exact flip in the spec names its family; otherwise the knobs' families, joined if they span several."""
    for f in spec.get("flips", []):
        if f["set"] == knobs:
            return f["family"]
    fam = knob_families(spec)
    return "+".join(sorted({fam.get(k, f"unknown:{k}") for k in knobs})) or "base"


def capmap_verdict(knobs: dict) -> tuple[str, str]:
    """(family, verdict) of a compare's B side from capability_map.json; the worst verdict when it spans families."""
    if not knobs or not CAPMAP.exists():
        return ("base" if not knobs else "unmapped"), "unmapped"
    j = json.loads(CAPMAP.read_text(encoding="utf-8"))
    fam = family_of(knobs, dict(families={}, flips=j.get("flips_spec", [])))
    if fam.startswith("unknown") or fam == "base" or "+" in fam or fam not in j["verdicts"]:
        fams = sorted({j["knobs"].get(k, f"unknown:{k}") for k in knobs})
        fam = "+".join(fams)
        vs = [j["verdicts"].get(f, "BLIND") for f in fams]
        return fam, min(vs, key=ORDER.index)
    return fam, j["verdicts"][fam]


def flip_status(deltas: list[float], lb) -> str:
    if lb is None:
        return "no key"
    if any(d == 0 for d in deltas):
        return "zero"
    s = {np.sign(d) == np.sign(lb) for d in deltas}
    return "match" if s == {True} else "mismatch" if s == {False} else "regime-dependent"


def family_verdict(flips: list[tuple[str, str]], lam_ok: bool, sanity_ok: bool) -> str:
    keyed = [(t, s) for t, s in flips if t in KEYED]
    ev = [s for t, s in keyed if t in EVIDENCE]
    if sum(s == "mismatch" for _, s in keyed) > sum(s == "match" for _, s in keyed):
        return "BLIND"
    if len(ev) >= 3 and all(s == "match" for s in ev):
        return "TRUSTED"
    if any(s == "match" for s in ev) and not any(s in ("mismatch", "regime-dependent") for s in ev) \
            and lam_ok and sanity_ok:
        return "TRUSTED-PROVISIONAL"
    return "ADVISORY"


def coupling_rate(r_set: np.ndarray, r0: np.ndarray, leak: np.ndarray) -> float:
    """Share of leak-hit molecules (BASE rank 1 == truth) whose final truth rank moves under a config."""
    return float(np.mean(r_set[leak] != r0[leak])) if leak.any() else 0.0


def coupled_verdict(verdict: str, coupling: float) -> str:
    """A family whose effect runs through BASE's leaked rank 1 can't be judged on val (BLIND stays BLIND)."""
    return "BASE-COUPLED" if coupling > COUPLING_TH and verdict != "BLIND" else verdict


def sweep_ok(values: list, mrrs: list[float], lb_opt) -> bool:
    i, j = int(np.argmax(mrrs)), values.index(lb_opt)
    return abs(i - j) <= 1


def _channels(state, mids, tk, cp) -> dict:
    def pos(lst, t):
        return next((i for i, s in enumerate(lst[:60], 1) if cp.canonicalize_smiles_to_ik14(s) == t), 0)
    out = {}
    for name, get in (("BASE", lambda m: list(state["base"][m][0])),
                      ("Engine 2", lambda m: ((state.get("eng") or {}).get(m) or {}).get("smiles") or []),
                      ("PubChem", lambda m: ((state.get("pc") or {}).get(m) or {}).get("pc") or [])):
        r = np.array([pos(get(m), tk[m]) for m in mids])
        out[name] = dict(top1=float(np.mean(r == 1)), in_list=float(np.mean(r > 0)))
    return out


def suite(shards: list[str], spec: dict | None = None, out_dir: Path = HERE, report: Path | None = None,
          eng_off: bool = False) -> dict:
    spec = spec or load_spec()
    T = truths()
    res = dict(shards=shards, mrr={}, sanity={}, channels={}, flips={}, sweeps={}, families={}, strata={})
    flip_d = {f["id"]: [] for f in spec["flips"]}
    lam_ok = True
    for shard in shards:
        state, deps, cp = load_dump(shard)
        mids = [m for m in state["te_mids"] if m in T]
        tk = {m: cp.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in mids}
        base = {"USE_ENG": False} if eng_off else {}

        def ranks(over, post=None):
            out = cp.run(state, deps, **{**base, **over})
            if post:
                out = post(out)
            return np.array([rank_of(out.get(m, []), tk[m], cp) for m in mids])
        r0 = ranks({})
        mrr0 = float(rr(r0).mean())
        res["mrr"][shard] = mrr0
        rng = np.random.default_rng(0)
        res["sanity"][shard] = dict(
            identity=float((rr(ranks({})) - rr(r0)).mean()),
            oracle=float((rr(ranks({}, lambda o: {m: [T[m]["smiles"]] + list(o.get(m, [])) for m in mids})) - rr(r0)).mean()),
            shuffle=float((rr(ranks({}, lambda o: {m: list(rng.permutation(list(o.get(m, [])))) for m in mids})) - rr(r0)).mean()))
        res["channels"][shard] = _channels(state, mids, tk, cp)
        leak = np.array([bool(state["base"][m][0]) and cp.canonicalize_smiles_to_ik14(state["base"][m][0][0]) == tk[m]
                         for m in mids])
        for f in spec["flips"]:
            if eng_off and "USE_ENG" in f["set"]:
                continue
            rf = ranks(f["set"])
            d = rr(rf) - rr(r0)
            lo, hi = bootstrap(d)
            flip_d[f["id"]].append(float(d.mean()))
            res["flips"].setdefault(f["id"], dict(family=f["family"], tier=f["tier"], lb=f["lb"], set=f["set"],
                                                  per_shard={}))["per_shard"][shard] = dict(
                delta=float(d.mean()), ci90=[lo, hi], up=int((d > 0).sum()), down=int((d < 0).sum()),
                coupling=coupling_rate(rf, r0, leak),
                delta_base_top1_truth=float(d[leak].mean()) if leak.any() else float("nan"),
                delta_base_top1_other=float(d[~leak].mean()) if (~leak).any() else float("nan"))
        res["strata"][shard] = dict(base_top1_truth=int(leak.sum()), other=int((~leak).sum()))
        for sw in spec.get("sweeps", []):
            rs = [ranks({sw["knob"]: v}) for v in sw["values"]]
            mrrs = [float(rr(r).mean()) for r in rs]
            ok = sweep_ok(sw["values"], mrrs, sw["lb_opt"]) if sw.get("evidence") else None
            if ok is False:
                lam_ok = False
            res["sweeps"].setdefault(sw["knob"], dict(values=sw["values"], evidence=bool(sw.get("evidence")),
                                                      lb_opt=sw.get("lb_opt"), per_shard={}))["per_shard"][shard] = dict(
                mrr=mrrs, max_abs_delta=float(max(abs(x - mrr0) for x in mrrs)), ok=ok,
                coupling=[coupling_rate(r, r0, leak) for r in rs])
        save_caches()
    sanity_ok = all(s["identity"] == 0 and abs(s["oracle"] - (1 - res["mrr"][sh])) < 1e-9 and s["shuffle"] < 0
                    for sh, s in res["sanity"].items())
    for fid, f in res["flips"].items():
        f["status"] = flip_status(flip_d[fid], f["lb"])
    for fam, knobs in spec["families"].items():
        fl = [(f["tier"], f["status"]) for f in res["flips"].values() if f["family"] == fam]
        cpl = [p["coupling"] for f in res["flips"].values() if f["family"] == fam for p in f["per_shard"].values()]
        cpl += [c for k in knobs if k in res["sweeps"] for p in res["sweeps"][k]["per_shard"].values()
                for c in p["coupling"]]
        coupling = max(cpl, default=0.0)
        res["families"][fam] = dict(verdict=coupled_verdict(family_verdict(fl, lam_ok, sanity_ok), coupling),
                                    coupling=coupling, measured=bool(cpl),
                                    flips=[fid for fid, f in res["flips"].items() if f["family"] == fam])
    res["lam_ok"], res["sanity_ok"], res["eng_off"] = lam_ok, sanity_ok, eng_off
    _write_capmap(res, spec, out_dir, report)
    return res


def _write_capmap(res: dict, spec: dict, out_dir: Path, report: Path | None) -> None:
    import datetime as dt
    sh = res["shards"]
    j = dict(generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), shards=sh, eng_off=res["eng_off"],
             knobs=knob_families(spec), verdicts={k: v["verdict"] for k, v in res["families"].items()},
             flips_spec=[dict(set=f["set"], family=f["family"]) for f in spec["flips"]], result=res)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_map.json").write_text(json.dumps(j, indent=1, default=float), encoding="utf-8")
    fmt = lambda x: f"{x:+.4f}" if isinstance(x, float) and np.isfinite(x) else "-"   # noqa: E731
    L = [f"# Harness capability map (GENERATED by `harness.py suite`; do not edit, {j['generated']})", "",
         f"Dumps: {', '.join(sh)}{' (Engine 2 off in every config)' if res['eng_off'] else ''}. Rules: "
         "gate_report.md 2026-10-11 'back-test suite' + 2026-10-12 'Option 2 BASE-coupling rule'. TRUSTED / "
         "TRUSTED-PROVISIONAL: the harness may decide. ADVISORY: the harness ranks ideas, the LB decides. BLIND: do not "
         "cite harness evidence. BASE-COUPLED: the effect runs through BASE's leaked rank 1 (coupling = max share of "
         f"leak-hit molecules whose truth rank moves > {COUPLING_TH}); harness evidence not citable, the LB decides.", "",
         "| family | verdict | coupling | flips | knobs |", "|---|---|---|---|---|"]
    for fam, v in res["families"].items():
        c = f"{v['coupling']:.3f}" if v.get("measured") else "not measured"
        L.append(f"| {fam} | **{v['verdict']}** | {c} | {', '.join(v['flips']) or '-'} | "
                 f"{', '.join(spec['families'][fam])} |")
    L += ["", f"Global: λ sweep {'PASS' if res['lam_ok'] else 'FAIL'}, mechanical sanity "
          f"{'PASS' if res['sanity_ok'] else 'FAIL'}.", "", "## Flips", "",
          "| flip | family | tier | LB | " + " | ".join(f"Δ {s} [90% CI] up/down, coupling" for s in sh) + " | status |",
          "|---|---|---|---|" + "---|" * len(sh) + "---|"]
    for fid, f in res["flips"].items():
        cells = [f"{fmt(p['delta'])} [{fmt(p['ci90'][0])}, {fmt(p['ci90'][1])}] {p['up']}/{p['down']}, {p['coupling']:.3f}"
                 for p in (f["per_shard"].get(s) for s in sh) if p]
        L.append(f"| `{fid}` | {f['family']} | {f['tier']} | {fmt(f['lb']) if f['lb'] is not None else '-'} | "
                 + " | ".join(cells) + f" | {f['status']} |")
    L += ["", "## Diagnostic: Δ split by whether BASE's rank 1 is the truth (leak-hit) or not", "",
          "| flip | " + " | ".join(f"{s} leak-hit (n {res['strata'][s]['base_top1_truth']}) | {s} other "
                                     f"(n {res['strata'][s]['other']})" for s in sh) + " |",
          "|---|" + "---|---|" * len(sh)]
    for fid, f in res["flips"].items():
        L.append(f"| `{fid}` | " + " | ".join(
            f"{fmt(f['per_shard'][s]['delta_base_top1_truth'])} | {fmt(f['per_shard'][s]['delta_base_top1_other'])}"
            for s in sh if s in f["per_shard"]) + " |")
    L += ["", "## Sweeps (MRR@25 per value; evidence sweeps must peak within one grid step of the LB value)", "",
          "| knob | values | evidence | " + " | ".join(f"{s} MRR (max abs Δ; max coupling)" for s in sh) + " | ok |",
          "|---|---|---|" + "---|" * len(sh) + "---|"]
    for k, w in res["sweeps"].items():
        cells = [" / ".join(f"{x:.3f}" for x in w["per_shard"][s]["mrr"])
                 + f" ({w['per_shard'][s]['max_abs_delta']:.3f}; {max(w['per_shard'][s]['coupling']):.3f})"
                 for s in sh if s in w["per_shard"]]
        oks = [w["per_shard"][s]["ok"] for s in sh if s in w["per_shard"]]
        L.append(f"| {k} | {w['values']} | {'yes, LB ' + str(w['lb_opt']) if w['evidence'] else 'no'} | "
                 + " | ".join(cells) + f" | {'-' if not w['evidence'] else ('PASS' if all(oks) else 'FAIL')} |")
    L += ["", "## Mechanical sanity", "", "| dump | MRR@25 | identity Δ | oracle Δ (= 1 − MRR) | shuffle Δ |",
          "|---|---|---|---|---|"]
    for s in sh:
        x = res["sanity"][s]
        L.append(f"| {s} | {res['mrr'][s]:.4f} | {x['identity']:+.4f} | {x['oracle']:+.4f} | {x['shuffle']:+.4f} |")
    L += ["", "## Channels (truth top-1 / anywhere in the list)", "", "| dump | " + " | ".join(
        ("BASE", "Engine 2", "PubChem")) + " |", "|---|---|---|---|"]
    for s in sh:
        c = res["channels"][s]
        L.append(f"| {s} | " + " | ".join(f"{c[k]['top1']:.3f} / {c[k]['in_list']:.3f}"
                                          for k in ("BASE", "Engine 2", "PubChem")) + " |")
    md = "\n".join(L) + "\n"
    (out_dir / "capability_map.md").write_text(md, encoding="utf-8")
    if report:
        head = f"\n### {j['generated']}: back-test suite result ({', '.join(sh)}{', Engine 2 off' if res['eng_off'] else ''})\n"
        body = "\n".join(L[L.index("| family | verdict | coupling | flips | knobs |"):L.index("## Flips")])
        with open(report, "a", encoding="utf-8") as f:
            f.write(head + body + "\nFull tables: research/harness/capability_map.md\n")


# ---------------------------------------------------------------- commands
def cmd_fetch(idea: str) -> int:
    name = Path(idea).name
    meta = dict(ln.split(":", 1) for ln in (REPO / idea / "STATUS").read_text().splitlines()[1:] if ":" in ln)
    kernel = meta["kernel"].strip()
    out = DUMPS / SHARD_OF.get(name, name)
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(out), "--file-pattern", r"post_dump"],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    ok = (out / "post_dump" / "state.pkl").exists()
    print("fetched" if ok else "state.pkl missing", out)
    return 0 if ok else 2


def cmd_lossmap(shards: list[str]) -> int:
    T = truths()
    lines = ["# Loss map of the 0.433 base (harness replay, unchanged config)", "",
             "Where the base loses MRR@25 on each slice. Levels read high (L07, gate_report.md); use the buckets to aim ideas.", ""]
    for shard in shards:
        state, deps, cp = load_dump(shard)
        mids = [m for m in state["te_mids"] if m in T]
        tk = {m: cp.canonicalize_smiles_to_ik14(T[m]["smiles"]) for m in mids}
        out = cp.run(state, deps)
        rk = np.array([rank_of(out.get(m, []), tk[m], cp) for m in mids])
        # where is the truth before the final list? anywhere in BASE top-60, PubChem list, Engine 2 top-40
        def anywhere(m):
            b = state["base"].get(m, ([],))[0]
            p = ((state.get("pc") or {}).get(m) or {}).get("pc") or []
            e = ((state.get("eng") or {}).get(m) or {}).get("smiles") or []
            return any(cp.canonicalize_smiles_to_ik14(s) == tk[m] for s in list(b) + list(p) + list(e))
        seen = np.array([anywhere(m) for m in mids])
        lines += [f"## {shard} (n = {len(mids)}, MRR@25 {rr(rk).mean():.3f})", "",
                  "| bucket | molecules | share | MRR headroom if fixed to rank 1 |", "|---|---|---|---|"]
        buckets = [("rank 1", rk == 1), ("rank 2-5", (rk >= 2) & (rk <= 5)), ("rank 6-25", rk >= 6),
                   ("not in top 25, but in some channel list", (rk == 0) & seen),
                   ("in no channel list (pool / retrieval miss)", (rk == 0) & ~seen)]
        for name, m in buckets:
            head = float((1 - rr(rk[m])).sum() / len(mids))
            lines.append(f"| {name} | {int(m.sum())} | {m.mean():.1%} | {head:+.3f} |")
        lines += ["", "| class | n | MRR |", "|---|---|---|"]
        for c in sorted({T[m]["cls"] for m in mids}):
            sel = np.array([T[m]["cls"] == c for m in mids])
            lines.append(f"| {c} | {int(sel.sum())} | {rr(rk[sel]).mean():.3f} |")
        lines.append("")
    (HERE / "loss_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def cmd_backtest() -> int:
    rows, ok = [], 0
    for flip, want in BACKTEST:
        r = compare("C2", {}, parse_set(flip))
        hit = np.sign(r["delta"]) == want
        ok += int(hit)
        rows.append(f"| `{flip}` | {r['delta']:+.4f} | [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}] | "
                    f"{r['up']}/{r['down']} | {r['delta_no_oracle']:+.4f} | {'yes' if hit else 'NO'} |")
    verdict = "PASS" if ok >= 3 else "FAIL"
    block = ["", f"### H2 back-test on C2 ({verdict}: {ok} of {len(BACKTEST)} signs match)", "",
             "| flip | Δ MRR@25 | 90% CI | up/down | Δ without FP-oracle | sign as LB |", "|---|---|---|---|---|---|", *rows]
    with open(HERE / "gate_report.md", "a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")
    print("\n".join(block))
    return 0 if ok >= 3 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("idea")
    c = sub.add_parser("compare"); c.add_argument("--dump", required=True); c.add_argument("--a", default="")
    c.add_argument("--b", required=True); c.add_argument("--post", type=Path); c.add_argument("--json", type=Path)
    lm = sub.add_parser("lossmap"); lm.add_argument("--dump", action="append", required=True)
    sub.add_parser("backtest")
    w = sub.add_parser("warm"); w.add_argument("--dump", required=True); w.add_argument("--procs", type=int, default=7)
    s = sub.add_parser("suite"); s.add_argument("--dump", action="append", required=True)
    s.add_argument("--eng-off", action="store_true"); s.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # tables print Δ; Windows consoles default to cp1252
    try:
        return _main(a)
    finally:
        save_caches()


def _main(a) -> int:
    if a.cmd == "suite":
        r = suite(a.dump, eng_off=a.eng_off, report=None if a.no_report else HERE / "gate_report.md")
        print((HERE / "capability_map.md").read_text(encoding="utf-8"))
        return 0 if r["sanity_ok"] else 2
    if a.cmd == "warm":
        return cmd_warm(a.dump, a.procs)
    if a.cmd == "fetch":
        return cmd_fetch(a.idea)
    if a.cmd == "lossmap":
        return cmd_lossmap(a.dump)
    if a.cmd == "backtest":
        return cmd_backtest()
    r = compare(a.dump, parse_set(a.a), parse_set(a.b), a.post)
    print(json.dumps(r, indent=1))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
