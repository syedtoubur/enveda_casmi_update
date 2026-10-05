"""R0 pipeline: per molecule, centred window -> lv (library), fr (MetFrag-lite), sv (ICEBERG) -> two-tier ranking.

Pass 1 plans every deep molecule and collects simulation requests; `sim.simulate` runs them under a deadline;
pass 2 scores and fills 25 slots (stage A tight window, B wider tiers append-only, C mass-nearest top-up).
Everything is injected (library, pool, worker, frag function), so the validation harness runs the same code.
"""
import time

import numpy as np

from .chem import neutral_mass, query_target
from .library import lib_sim
from .metfrag import frag_scores
from .rank import Slots, rank_two_tier
from .sim import build_requests, sim_priority, simulate
from .spectra import ce_tuple, prep, spectrum_ces, spectrum_sv


def scored_range(pool, target, cfg):
    """Contiguous pool range scored by stages A + B: +-ppm_win (+-ppm_fallback if empty), widened through
    ppm_expand until >= topn candidates."""
    lo, hi = pool.bounds(target, cfg.ppm_win)
    if hi == lo:
        lo, hi = pool.bounds(target, cfg.ppm_fallback)
    if hi - lo < cfg.topn:
        for ppm in cfg.ppm_expand[1:]:
            a, b = pool.bounds(target, ppm)
            lo, hi = min(lo, a), max(hi, b)
            if hi - lo >= cfg.topn:
                break
    return lo, hi


def molecule_target(sub, cfg):
    nm = neutral_mass(sub.precursor_mz.to_numpy(np.float64), sub.adduct.to_numpy(dtype=object))
    return query_target(nm, cfg.ppm_center)


def plan_molecule(sub, target, lib, pool, cfg, sim_tier1=False, frag_fn=frag_scores, lib_exclude=None):
    """Pass 1 for one molecule: lv, fr on the scored range; covered query spectra; which candidates to simulate."""
    specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
    lib_hits = lib_sim(lib, specs, target, cfg, exclude=lib_exclude)
    ion_mode = float(np.mean([1.0 if m == "positive" else -1.0 for m in sub.ionization_mode]))
    lo, hi = scored_range(pool, target, cfg)
    cand = np.arange(lo, hi)
    lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
    fr = frag_fn([pool.smiles[c] for c in cand], specs, ion_mode, cfg) if len(cand) else np.zeros(0, np.float32)
    ppm = np.abs(pool.mass[cand] / target - 1.0) * 1e6

    qs, n_ce_default = [], 0
    for r in sub.itertuples():
        if r.adduct not in cfg.covered:
            continue
        ces, defaulted = spectrum_ces(ce_tuple(r.collision_energy_ev), cfg.merge_rule, cfg.ce_default)
        n_ce_default += defaulted
        q = prep(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), cfg)
        if len(q[0]):
            qs.append((r.adduct, ces, q))
    conds = sorted({(a, c) for a, ces, _ in qs for c in ces})

    valid = np.array([k is not None and len(str(k)) >= 14 for k in pool.keys[cand]], bool)
    to_sim = valid & ((lv < cfg.tau) | sim_tier1) & bool(cfg.use_sim)
    return dict(target=target, lib_hits=lib_hits, ion_mode=ion_mode, specs=specs,
                fr={int(c): float(f) for c, f in zip(cand, fr)}, qs=qs, conds=conds, n_cand=len(cand),
                n_ce_default=n_ce_default, sim_idx=cand[to_sim],
                prio=sim_priority(lv[to_sim], fr[to_sim], ppm[to_sim]))


def features(plan, idx, pool, sim, cfg, frag_fn=frag_scores):
    """lv, sv, fr, ppm for pool indices idx (fr computed now for indices pass 1 did not score)."""
    lv = np.array([plan["lib_hits"].get(pool.keys[c], 0.0) for c in idx], np.float32)
    miss = [int(c) for c in idx if int(c) not in plan["fr"]]
    if miss:
        plan["fr"].update(zip(miss, map(float, frag_fn([pool.smiles[c] for c in miss], plan["specs"],
                                                       plan["ion_mode"], cfg))))
    fr = np.array([plan["fr"][int(c)] for c in idx], np.float32)
    ppm = np.abs(pool.mass[idx] / plan["target"] - 1.0) * 1e6
    sv = np.zeros(len(idx), np.float32)
    for j, c in enumerate(idx):
        k = pool.keys[c]
        for adduct, ces, q in plan["qs"]:
            sv[j] = max(sv[j], spectrum_sv(q, [sim.get((k, adduct, ce)) for ce in ces], cfg))
    return lv, sv, fr, ppm


def rank_molecule(plan, pool, sim, cfg, frag_fn=frag_scores):
    """Pass 2 for one deep molecule -> (Slots, diag dict, stage-A features)."""
    target, slots = plan["target"], Slots(cfg.topn)
    cand = pool.window(target, cfg.ppm_win)
    if len(cand) == 0:
        cand = pool.window(target, cfg.ppm_fallback)
    diag, stage_a = None, None
    if len(cand):
        lv, sv, fr, ppm = features(plan, cand, pool, sim, cfg, frag_fn)
        order = rank_two_tier(lv, sv, fr, ppm, cfg.tau)
        slots.take(pool, cand[order])
        diag = dict(n_cand=len(cand), n_covered_spec=len(plan["qs"]), n_cond=len(plan["conds"]),
                    lv_max=float(lv.max()), sv_max=float(sv.max()), fr_max=float(fr.max()),
                    top_is_tier1=bool(lv[order[0]] >= cfg.tau))
        stage_a = (cand, lv, sv, fr, ppm)
    if cfg.expand_backfill and not slots.full:
        for tier in cfg.ppm_expand[1:]:
            wide = pool.window(target, tier)
            new = wide[~np.isin(wide, cand)]
            if len(new):
                slots.take(pool, new[rank_two_tier(*features(plan, new, pool, sim, cfg, frag_fn), cfg.tau)])
            if slots.full:
                break
    if not slots.full:
        fill_mass_nearest(slots, pool, target, cfg.backfill_ppm)
    return slots, diag, stage_a


def fill_mass_nearest(slots, pool, target, ppm):
    wide = pool.window(target, ppm)
    if len(wide):
        slots.take(pool, wide[np.argsort(np.abs(pool.mass[wide] - target), kind="stable")])


def run(mols, lib, pool, cfg, call=None, deep=None, t0=None, strict=False, frag_fn=frag_scores, lib_exclude=None,
        log=lambda s: print(s, flush=True)):
    """Rank every molecule.

    mols: [(molecule_id, sub_df)]. deep: molecule ids that get the full engine (None = all); the rest get
    mass-nearest +-25 ppm candidates. call: IcebergWorker (None = no simulation). lib_exclude: optional
    {molecule_id: boolean mask over library spectra} (validation bans).
    Returns ({molecule_id: [25 smiles]}, report dict, plans).
    """
    t0 = time.time() if t0 is None else t0
    times, plans, out, diags, stage_a = {}, {}, {}, {}, {}
    sim_tier1 = cfg.sim_tier1 or strict

    t = time.time()
    for i, (mid, sub) in enumerate(mols):
        target = molecule_target(sub, cfg)
        if not np.isfinite(target) or (deep is not None and mid not in deep):
            plans[mid] = dict(target=target, deep=False)
            continue
        excl = None if lib_exclude is None else lib_exclude.get(mid)
        plans[mid] = dict(plan_molecule(sub, target, lib, pool, cfg, sim_tier1, frag_fn, excl), deep=True)
        if i % 50 == 0:
            log(f"  pass 1: {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    deep_ids = [m for m, p in plans.items() if p["deep"]]
    times["pass1_s"] = round(time.time() - t, 1)

    req = build_requests(({"key": pool.keys[p["sim_idx"]], "smiles": pool.smiles[p["sim_idx"]],
                           "mass": pool.mass[p["sim_idx"]], "prio": p["prio"], "conds": p["conds"]}
                          for p in (plans[m] for m in deep_ids)), cfg.instrument, cfg.ce_mode)
    log(f"[pass 1] {len(deep_ids):,} deep molecules in {times['pass1_s']:.0f}s; {len(req):,} simulation requests")

    t = time.time()
    if call is not None and len(req):
        reserve = cfg.pass2_s_per_mol * len(deep_ids) + 300.0
        sim, sim_stats = simulate(req, call, t0 + cfg.hard_budget_h * 3600.0 - reserve, cfg, strict=strict, log=log)
    else:
        sim, sim_stats = {}, dict(n_requests=len(req), n_predicted=0, sim_ok_share=float("nan"), sim_errors={},
                                  failed_rows=0, capped_rows=len(req) if call is None else 0)
    times["sim_s"] = round(time.time() - t, 1)

    t = time.time()
    for mid, _ in mols:
        p = plans[mid]
        if p["deep"]:
            slots, diag, sa = rank_molecule(p, pool, sim, cfg, frag_fn)
            if diag is not None:
                diags[mid], stage_a[mid] = diag, sa
        else:
            slots = Slots(cfg.topn)
            if np.isfinite(p["target"]):
                fill_mass_nearest(slots, pool, p["target"], 25.0)
        out[mid] = slots.padded()
    times["pass2_s"] = round(time.time() - t, 1)

    report = dict(times=times, n_molecules=len(mols), n_deep=len(deep_ids),
                  ce_defaulted=int(sum(plans[m]["n_ce_default"] for m in deep_ids)), **sim_stats,
                  tier1_rate=float(np.mean([d["top_is_tier1"] for d in diags.values()])) if diags else float("nan"),
                  tier2_plumbing=tier2_plumbing(stage_a, plans, cfg))
    return out, report, dict(plans=plans, diags=diags, stage_a=stage_a, sim=sim)


def tier2_plumbing(stage_a, plans, cfg, n=20):
    """Zero the tier-1 winner's lv (as if it had no real spectrum): does sv alone keep it in the top 25?"""
    pos = []
    for mid in list(stage_a)[:n]:
        cand, lv, sv, fr, ppm = stage_a[mid]
        top = rank_two_tier(lv, sv, fr, ppm, cfg.tau)[0]
        if lv[top] < cfg.tau or not plans[mid]["qs"]:
            continue
        lv2 = lv.copy()
        lv2[top] = 0.0
        pos.append(int(np.flatnonzero(rank_two_tier(lv2, sv, fr, ppm, cfg.tau) == top)[0]) + 1)
    return dict(n=len(pos), in_top25=float(np.mean(np.array(pos) <= 25)) if pos else None,
                median_rank=float(np.median(pos)) if pos else None)
