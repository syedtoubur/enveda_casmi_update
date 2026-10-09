"""R0 pipeline: per molecule, centred window -> lv (library), fr (MetFrag-lite), sv (ICEBERG) -> two-tier ranking.

Pass 1 plans every deep molecule and collects simulation requests; `sim.simulate` runs them under a deadline;
pass 2 scores and fills 25 slots (stage A tight window, B wider tiers append-only, C mass-nearest top-up).
Everything is injected (library, pool, worker, frag function), so the validation harness runs the same code.
"""
import time

import numpy as np

from .analog import analog_hits, analog_set, class_vote, molecule_analogs, vote_bonus, vote_conf
from .chem import neutral_mass, query_target
from .library import lib_sim
from .fpnet import molecule_spectra
from .metfrag import frag_scores
from .pool import F_TRAIN
from .rank import Slots, fp_rrf, rank_two_tier
from .sim import build_requests, sim_adducts, sim_priority, simulate
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


def molecule_vote(lib, specs, target, ion_mode, cfg, vote_label, lib_exclude=None):
    """R2: NPClassifier superclass voted by the delta-list analog library hits -> (label or None, confidence)."""
    hits = analog_hits(lib, specs, target, cfg, "delta", polarity=1 if ion_mode >= 0 else -1, exclude=lib_exclude,
                       ppm=cfg.vote_ppm)
    win, w, _ = class_vote(hits, vote_label.get, cfg.vote_k, cfg.vote_min_sim)
    return win, vote_conf(w, win)


def plan_molecule(sub, target, lib, pool, cfg, sim_tier1=False, frag_fn=frag_scores, lib_exclude=None,
                  vote_label=None):
    """Pass 1 for one molecule: lv, fr on the scored range; covered query spectra; which candidates to simulate;
    the class vote (R2, when vote_label is given)."""
    specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
    lib_hits = lib_sim(lib, specs, target, cfg, exclude=lib_exclude)
    ion_mode = float(np.mean([1.0 if m == "positive" else -1.0 for m in sub.ionization_mode]))
    vote = molecule_vote(lib, specs, target, ion_mode, cfg, vote_label, lib_exclude)         if cfg.class_vote and vote_label is not None else (None, 0.0)
    an_hits, an_s = None, 0.0
    if cfg.analog_channel:                  # R5: open analogue hits (top few hundred kept; the profile is built later)
        ta = time.time()
        pols = [1 if m == "positive" else -1 for m in sub.ionization_mode]
        an_hits = dict(analog_set(molecule_analogs(lib, specs, pols, target, cfg, lib_exclude), cfg.analog_smin,
                                  4 * max(cfg.analog_m, cfg.analog_k)))
        an_s = time.time() - ta
    lo, hi = scored_range(pool, target, cfg)
    cand = np.arange(lo, hi)
    lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
    fr = frag_fn([pool.smiles[c] for c in cand], specs, ion_mode, cfg) if len(cand) else np.zeros(0, np.float32)
    ppm = np.abs(pool.mass[cand] / target - 1.0) * 1e6

    qs, n_ce_default, adducts = [], 0, sim_adducts(cfg)
    for r in sub.itertuples():
        if r.adduct not in adducts:
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
                n_ce_default=n_ce_default, sim_idx=cand[to_sim], vote=vote, an_hits=an_hits, an_s=an_s,
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


def class_bonus(plan, idx, pool, cfg):
    """R2 tier-2 bonus for pool indices idx (None when the vote is off)."""
    if not cfg.class_vote or pool.npc is None:
        return None
    sc, sp, names = pool.npc
    return vote_bonus(sc[idx], sp[idx], names, *plan.get("vote", (None, 0.0)), cfg.vote_beta)


def as_banks(fp_bank):
    """None | one FPBank | a list of them -> list (R3 passes one bank, R4 one per family)."""
    return [] if fp_bank is None else (list(fp_bank) if isinstance(fp_bank, (list, tuple)) else [fp_bank])


def fp_scores(plan, idx, pool, cfg, fp_bank):
    """R3/R4 f.z for pool indices idx: a list with one array per bank (NaN where a bank has no logits or the SMILES
    does not parse), or None when the channel is off or no bank has logits for the molecule."""
    zs = plan.get("fpz")
    if not cfg.fp_kinds() or fp_bank is None or not zs or all(z is None for z in zs):
        return None
    keys, smis = [pool.keys[c] for c in idx], [pool.smiles[c] for c in idx]
    return [b.fz(z, keys, smis) for b, z in zip(as_banks(fp_bank), zs)]


def analog_scores(plan, idx, pool, cfg, analog_bank):
    """R5 analogue score for pool indices idx (NaN = not scored: train structures when cfg.analog_exclude_train,
    unless exempted in plan["an_free"], and structures without a fingerprint), or None when the channel is off or the
    molecule has fewer than analog_min_n analogues."""
    prof = plan.get("an_prof")
    if not cfg.analog_channel or analog_bank is None or prof is None or not len(idx):
        return None
    keys = [pool.keys[c] for c in idx]
    s = analog_bank.score(prof, keys, [pool.smiles[c] for c in idx]).astype(np.float64)
    if cfg.analog_exclude_train and pool.flags is not None:
        free = plan.get("an_free") or set()
        train = (np.asarray(pool.flags[idx], np.int64) & F_TRAIN) != 0
        s[train & ~np.array([k in free for k in keys], bool)] = np.nan
    return s


def rank_molecule(plan, pool, sim, cfg, frag_fn=frag_scores, fp_bank=None, analog_bank=None):
    """Pass 2 for one deep molecule -> (Slots, diag dict, stage-A features)."""
    target, slots = plan["target"], Slots(cfg.topn)
    cand = pool.window(target, cfg.ppm_win)
    if len(cand) == 0:
        cand = pool.window(target, cfg.ppm_fallback)
    diag, stage_a = None, None
    if len(cand):
        lv, sv, fr, ppm = features(plan, cand, pool, sim, cfg, frag_fn)
        bonus = class_bonus(plan, cand, pool, cfg)
        fz = fp_scores(plan, cand, pool, cfg, fp_bank)
        an = analog_scores(plan, cand, pool, cfg, analog_bank)
        order = rank_two_tier(lv, sv, fr, ppm, cfg.tau, bonus, fz, cfg.fp_rrf_k, cfg.fp_w(), an, cfg.analog_w)
        slots.take(pool, cand[order])
        diag = dict(n_cand=len(cand), n_covered_spec=len(plan["qs"]), n_cond=len(plan["conds"]),
                    lv_max=float(lv.max()), sv_max=float(sv.max()), fr_max=float(fr.max()),
                    top_is_tier1=bool(lv[order[0]] >= cfg.tau))
        if bonus is not None:
            plain = rank_two_tier(lv, sv, fr, ppm, cfg.tau)
            diag.update(voted=plan["vote"][0] is not None, bonus_max=float(bonus.max()),
                        reordered=not np.array_equal(order[:cfg.topn], plain[:cfg.topn]),
                        vote_first=bool(plan["vote"][0] is not None and sv.max() <= 0 and bonus.max() > 0))
        if cfg.fp_kinds():
            nofp = rank_two_tier(lv, sv, fr, ppm, cfg.tau, bonus)
            diag.update(fp_ok=fz is not None and any(bool(np.isfinite(f).any()) for f in fz),
                        fp_reordered=fz is not None and not np.array_equal(order[:cfg.topn], nofp[:cfg.topn]))
            if fz is not None and len(fz) > 1:          # R4: per family, does dropping it change the top 25?
                w = cfg.fp_w()
                diag.update(fp_ok_by=[bool(np.isfinite(f).any()) for f in fz],
                            fp_reordered_by=[not np.array_equal(order[:cfg.topn], rank_two_tier(
                                lv, sv, fr, ppm, cfg.tau, bonus, fz[:i] + fz[i + 1:], cfg.fp_rrf_k,
                                w[:i] + w[i + 1:])[:cfg.topn]) for i in range(len(fz))])
        if cfg.analog_channel:
            noan = rank_two_tier(lv, sv, fr, ppm, cfg.tau, bonus, fz, cfg.fp_rrf_k, cfg.fp_w())
            diag.update(an_ok=an is not None and bool(np.isfinite(an).any()),
                        an_scored=int(np.isfinite(an).sum()) if an is not None else 0,
                        an_reordered=not np.array_equal(order[:cfg.topn], noan[:cfg.topn]))
        stage_a = (cand, lv, sv, fr, ppm)
    if cfg.expand_backfill and not slots.full:
        for tier in cfg.ppm_expand[1:]:
            wide = pool.window(target, tier)
            new = wide[~np.isin(wide, cand)]
            if len(new):
                slots.take(pool, new[rank_two_tier(*features(plan, new, pool, sim, cfg, frag_fn), cfg.tau,
                                                   class_bonus(plan, new, pool, cfg),
                                                   fp_scores(plan, new, pool, cfg, fp_bank), cfg.fp_rrf_k,
                                                   cfg.fp_w(), analog_scores(plan, new, pool, cfg, analog_bank),
                                                   cfg.analog_w)])
            if slots.full:
                break
    if not slots.full:
        n0 = len(slots.keys)
        fp_fn = (lambda idx: fp_scores(plan, idx, pool, cfg, fp_bank)) if cfg.fp_stagec and cfg.fp_kinds() else None
        fill_mass_nearest(slots, pool, target, cfg.backfill_ppm, fp_fn, cfg.fp_w(), cfg.fp_rrf_k)
        if diag is not None:
            diag["stagec_slots"] = len(slots.keys) - n0
    return slots, diag, stage_a


def fill_mass_nearest(slots, pool, target, ppm, fp_fn=None, fp_w=None, rrf_k=3.0):
    """Stage C top-up: mass-nearest within +-ppm. R4: with fp_fn (pool indices -> list of f.z or None) the window is
    ordered by the fused FP key sum w_i / (k + rank(f.z_i)) first, |delta mass| second."""
    wide = pool.window(target, ppm)
    if not len(wide):
        return
    dm = np.abs(pool.mass[wide] - target)
    fz = fp_fn(wide) if fp_fn is not None else None
    order = np.lexsort((dm, -fp_rrf(fz, fp_w, rrf_k))) if fz is not None else np.argsort(dm, kind="stable")
    slots.take(pool, wide[order])


def run(mols, lib, pool, cfg, call=None, deep=None, t0=None, strict=False, frag_fn=frag_scores, lib_exclude=None,
        vote_label=None, fp_bank=None, analog_bank=None, analog_free=None, log=lambda s: print(s, flush=True)):
    """Rank every molecule.

    mols: [(molecule_id, sub_df)]. deep: molecule ids that get the full engine (None = all); the rest get
    mass-nearest +-25 ppm candidates. call: IcebergWorker (None = no simulation). lib_exclude: optional
    {molecule_id: boolean mask over library spectra} (validation bans). vote_label: {library key: superclass} for
    the R2 class vote (cfg.class_vote; the pool must carry NPClassifier labels). fp_bank: fpnet.FPBank for the R3
    FPNet channel (cfg.fp_channel), or a list of banks aligned with cfg.fp_banks (R4 family fusion); logits are
    computed in pass 1, then the GPU is released before simulation. analog_bank: analog.AnalogBank for the R5
    analogue channel (cfg.analog_channel); analog_free: optional {molecule_id: keys} that the analogue list scores
    even though they are train structures (validation: the held-out truth).
    Returns ({molecule_id: [25 smiles]}, report dict, plans).
    """
    t0 = time.time() if t0 is None else t0
    times, plans, out, diags, stage_a = {}, {}, {}, {}, {}
    sim_tier1 = cfg.sim_tier1 or strict
    if cfg.class_vote and (vote_label is None or pool.npc is None):
        raise ValueError("class_vote needs vote_label and a pool loaded with_npc=True")
    banks = as_banks(fp_bank)
    if len(banks) != len(cfg.fp_kinds()):
        raise ValueError(f"fp banks {[b.kind for b in banks]} do not match cfg.fp_kinds() {cfg.fp_kinds()}")
    if cfg.analog_channel and analog_bank is None:
        raise ValueError(f"analog_channel={cfg.analog_channel!r} needs an analog_bank")

    t = time.time()
    for i, (mid, sub) in enumerate(mols):
        target = molecule_target(sub, cfg)
        if not np.isfinite(target) or (deep is not None and mid not in deep):
            plans[mid] = dict(target=target, deep=False)
            continue
        excl = None if lib_exclude is None else lib_exclude.get(mid)
        plans[mid] = dict(plan_molecule(sub, target, lib, pool, cfg, sim_tier1, frag_fn, excl, vote_label),
                          deep=True)
        if cfg.analog_channel:
            ta = time.time()
            plans[mid]["an_prof"] = analog_bank.profile(plans[mid].pop("an_hits") or {})
            plans[mid]["an_free"] = set((analog_free or {}).get(mid, ()))
            times["analog_s"] = times.get("analog_s", 0.0) + plans[mid]["an_s"] + time.time() - ta
        if banks:
            tf, spectra = time.time(), molecule_spectra(sub)
            plans[mid]["fpz"] = [b.molecule_z(spectra) for b in banks]
            times["fp_z_s"] = times.get("fp_z_s", 0.0) + time.time() - tf
        if i % 50 == 0:
            log(f"  pass 1: {i + 1}/{len(mols)} | {time.time() - t0:.0f}s")
    deep_ids = [m for m, p in plans.items() if p["deep"]]
    times["pass1_s"] = round(time.time() - t, 1)
    if "analog_s" in times:
        times["analog_s"] = round(times["analog_s"], 1)
    if banks:
        times["fp_z_s"] = round(times.get("fp_z_s", 0.0), 1)
        for b in banks:
            b.release()

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
            slots, diag, sa = rank_molecule(p, pool, sim, cfg, frag_fn, fp_bank, analog_bank)
            if diag is not None:
                diags[mid], stage_a[mid] = diag, sa
        else:
            slots = Slots(cfg.topn)
            if np.isfinite(p["target"]):
                fill_mass_nearest(slots, pool, p["target"], 25.0)
        out[mid] = slots.padded()
    times["pass2_s"] = round(time.time() - t, 1)

    report = dict(times=times, n_molecules=len(mols), n_deep=len(deep_ids), sim_model=cfg.sim_model,
                  n_covered_spec=int(sum(len(plans[m]["qs"]) for m in deep_ids)),
                  ce_defaulted=int(sum(plans[m]["n_ce_default"] for m in deep_ids)), **sim_stats,
                  cand_median=float(np.median([plans[m]["n_cand"] for m in deep_ids])) if deep_ids else float("nan"),
                  tier1_rate=float(np.mean([d["top_is_tier1"] for d in diags.values()])) if diags else float("nan"),
                  tier2_plumbing=tier2_plumbing(stage_a, plans, cfg),
                  class_vote=cfg.class_vote, **vote_counters(diags),
                  fp_channel=cfg.fp_channel, fp_banks=list(cfg.fp_kinds()), fp_weights=list(cfg.fp_w()),
                  **fp_counters(diags, plans, deep_ids),
                  analog_channel=cfg.analog_channel, **analog_counters(diags, plans, deep_ids),
                  fp_bank=(banks[0].describe() if len(banks) == 1 else [b.describe() for b in banks]) if banks else None)
    return out, report, dict(plans=plans, diags=diags, stage_a=stage_a, sim=sim)


def vote_counters(diags):
    """R2 fire counters over the deep molecules (all 0 when the vote is off)."""
    d = [x for x in diags.values() if "voted" in x]
    return dict(n_vote_mol=int(sum(x["voted"] for x in d)), n_bonus_reorders=int(sum(x["reordered"] for x in d)),
                n_vote_first=int(sum(x["vote_first"] for x in d)))


def fp_counters(diags, plans, deep_ids):
    """R3 fire counters (0 when the channel is off): molecules with logits, with any f.z in stage A, and whose
    stage-A top 25 differs from the FP-free order."""
    d = [x for x in diags.values() if "fp_ok" in x]
    zs = [plans[m].get("fpz") or [] for m in deep_ids]
    out = dict(n_fp_z=int(sum(any(z is not None for z in zz) for zz in zs)),
               n_fp_mol=int(sum(x["fp_ok"] for x in d)), n_fp_reorders=int(sum(x["fp_reordered"] for x in d)),
               n_stagec_mol=int(sum(x.get("stagec_slots", 0) > 0 for x in diags.values())),
               n_stagec_slots=int(sum(x.get("stagec_slots", 0) for x in diags.values())))
    nb = max((len(zz) for zz in zs), default=0)
    if nb > 1:                                          # R4: the same counters per family (order = cfg.fp_banks)
        out.update(n_fp_z_by=[int(sum(len(zz) > i and zz[i] is not None for zz in zs)) for i in range(nb)],
                   n_fp_mol_by=[int(sum(x["fp_ok_by"][i] for x in d if "fp_ok_by" in x)) for i in range(nb)],
                   n_fp_reorders_by=[int(sum(x["fp_reordered_by"][i] for x in d if "fp_reordered_by" in x))
                                     for i in range(nb)])
    return out


def analog_counters(diags, plans, deep_ids):
    """R5 fire counters (0 when the channel is off): molecules with a profile / dropped (< analog_min_n analogues),
    with any scored candidate in stage A, whose stage-A top 25 the list changed; median analogues per profile."""
    d = [x for x in diags.values() if "an_ok" in x]
    profs = [plans[m].get("an_prof") for m in deep_ids if "an_prof" in plans[m]]
    ns = [p["n"] for p in profs if p is not None]
    return dict(n_analog_mol=len(ns), n_analog_dropped=sum(p is None for p in profs),
                n_analog_ok=int(sum(x["an_ok"] for x in d)), n_analog_reorders=int(sum(x["an_reordered"] for x in d)),
                analog_n_median=float(np.median(ns)) if ns else float("nan"))


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
