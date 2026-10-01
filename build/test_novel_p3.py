"""specs_novel P3 fixture tests: the model, the three searches, the data path, the metrics.

    python build/test_novel_p3.py

Plain asserts on synthetic data, no Drive and no GPU. The end-to-end notebook check is
`build/novel_p3_dryrun.py`, which runs the notebooks against a fake Drive.
"""
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("NOVEL_ALLOW_ANY_EMB", "1")

import selfies as sf
import torch
from rdkit import Chem

import novel_decoder as nd
import novel_p3_data as p3d
import novel_p3_eval as p3e
import simsim_common as ssc

SEED = 20260929
# neutral mass must land in novel_cohort.label_ok's [157, 1159] window
SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
          "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "CCCCCCCCCCCCCCCC(=O)O",
          "c1ccc2c(c1)ccc1ccccc12", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21"]


def vocab_and_table():
    alpha = set(sf.get_semantic_robust_alphabet())
    for s in SMILES:
        alpha |= set(sf.split_selfies(nd.prep_structure(s)[2]))
    v = nd.SelfiesVocab.from_alphabet(alpha)
    return v, nd.TokenTable(v)


VOCAB, TABLE = vocab_and_table()


def tiny(max_len=60, **kw):
    torch.manual_seed(SEED)
    return nd.DreamsSelfiesDecoder(len(VOCAB), pad_id=VOCAB.pad_id, sos_id=VOCAB.sos_id,
                                   eos_id=VOCAB.eos_id, d_model=kw.pop("d_model", 64),
                                   n_layers=kw.pop("n_layers", 2), n_heads=4, d_ff=128,
                                   dropout=0.0, max_len=max_len, **kw)


# --------------------------------------------------------------------------------------------- #

def check_constants():
    """The adduct list and CE bins must not drift from the rest of the project."""
    assert tuple(ssc.TEST_ADDUCTS) == nd.TEST_ADDUCTS, "adduct list drifted from simsim_common"
    assert nd.N_CE_BINS == len(nd.CE_BINS) + 2 == 6, nd.N_CE_BINS
    assert nd.CE_UNKNOWN == 5, nd.CE_UNKNOWN
    # ce_bin: the bin edges, and everything unusable -> unknown
    assert nd.ce_bin(10.0) == 0 and nd.ce_bin(20.0) == 1 and nd.ce_bin(60.0) == 4
    for bad in (None, float("nan"), float("inf"), -1.0, 0.0, "x", ""):
        assert nd.ce_bin(bad) == nd.CE_UNKNOWN, bad
    # formula_vector keeps hydrogen; TokenTable.counts_vector drops it. That difference is the
    # reason formula_vector exists, so pin it.
    counts = {"C": 9, "H": 8, "O": 4}
    fv = nd.formula_vector(counts)
    assert fv[nd.ALLOWED_ELEMENTS.index("H")] == 8, fv
    cv, missing = TABLE.counts_vector(counts)
    assert missing == 0
    print("  constants + ce_bin + formula_vector: OK")


def check_memory():
    m = tiny()
    k = 4
    embs = torch.randn(1, k, 1024)
    ad = torch.tensor([[0, 3, 6, 9]])
    ce = torch.tensor([[1, 5, 2, 0]])
    fv = torch.from_numpy(nd.formula_vector({"C": 6, "H": 6, "O": 1}))[None]
    mem, pad = m.build_memory(embs, ad, ce, fv)
    assert mem.shape == (1, k + 1, m.d_model), mem.shape
    assert pad.shape == (1, k + 1) and not bool(pad.any())

    m.eval()
    with torch.no_grad():
        # the spectra are a SET: permuting them must not change the memory as a multiset, and must
        # leave the formula token (always last) untouched. Catches an accidental positional encoding.
        perm = [2, 0, 3, 1]
        mem2, _ = m.build_memory(embs[:, perm], ad[:, perm], ce[:, perm], fv)
        a = mem[0, :k].sort(dim=0).values
        b = mem2[0, :k].sort(dim=0).values
        assert torch.allclose(a, b, atol=1e-5), "spectrum tokens are not permutation-invariant"
        assert torch.allclose(mem[0, k], mem2[0, k], atol=1e-6), "the formula token moved"

        # a padded spectrum is both masked and zeroed
        e4 = torch.cat([embs, torch.randn(1, 1, 1024)], 1)
        a4 = torch.cat([ad, torch.zeros(1, 1, dtype=torch.long)], 1)
        c4 = torch.cat([ce, torch.zeros(1, 1, dtype=torch.long)], 1)
        sm = torch.tensor([[True] * k + [False]])
        mem3, pad3 = m.build_memory(e4, a4, c4, fv, sm)
        assert pad3[0].tolist() == [False] * k + [True, False], pad3
        assert bool((mem3[0, k] == 0).all()), "a padded spectrum token was not zeroed"
        assert torch.allclose(mem3[0, k + 1], mem[0, k], atol=1e-6), "padding moved the formula token"

        # a different formula must change the formula token (it is actually conditioned on)
        fv2 = torch.from_numpy(nd.formula_vector({"C": 9, "H": 8, "O": 4}))[None]
        mem4, _ = m.build_memory(embs, ad, ce, fv2)
        assert not torch.allclose(mem4[0, k], mem[0, k], atol=1e-4), "the formula token is inert"
    print("  memory shapes, set-invariance, padding, formula conditioning: OK")


def check_tied_and_forward():
    m = tiny().eval()
    fv = torch.from_numpy(nd.formula_vector({"C": 6, "H": 6}))[None]
    mem, pad = m.build_memory(torch.randn(1, 2, 1024), torch.zeros(1, 2, dtype=torch.long),
                              torch.full((1, 2), 5), fv)
    ids = torch.tensor([[VOCAB.sos_id, VOCAB.tok2id["[C]"], VOCAB.tok2id["[C]"]]])
    with torch.no_grad():
        lg = m(ids, mem, pad)
    assert lg.shape == (1, 3, len(VOCAB)), lg.shape
    # the output projection IS the input embedding, so a change to one moves the other
    assert m.tok_emb.weight.shape[0] == len(VOCAB)
    with torch.no_grad():
        before = m(ids, mem, pad).clone()
        m.tok_emb.weight[5] += 1.0
        after = m(ids, mem, pad)
    assert not torch.allclose(before, after), "output is not tied to the input embedding"
    # causality: a later token cannot change an earlier position's logits
    m2 = tiny().eval()
    ids3 = torch.tensor([[VOCAB.sos_id, VOCAB.tok2id["[C]"], VOCAB.tok2id["[N]"]]])
    with torch.no_grad():
        a = m2(ids, mem, pad)[0, 0]
        b = m2(ids3, mem, pad)[0, 0]
    assert torch.allclose(a, b, atol=1e-6), "the decoder is not causal"
    print("  forward shape, tied embeddings, causality: OK")


def _train_one(m, can, formula, steps=250, lr=3e-3):
    """Nudge a tiny model towards one molecule, so the searches actually emit leaves to inspect.
    An untrained model almost never completes an exact formula, which would leave the invariant
    assertions below inspecting nothing."""
    ids = VOCAB.encode(nd.prep_structure(can)[2])
    fv = torch.from_numpy(nd.formula_vector(formula))[None]
    embs, ad, ce = torch.randn(1, 1, 1024), torch.zeros(1, 1, dtype=torch.long), torch.full((1, 1), 5)
    tgt = torch.tensor([[VOCAB.sos_id] + ids + [VOCAB.eos_id]])
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        mem, pad = m.build_memory(embs, ad, ce, fv)
        lg = m(tgt[:, :-1], mem, pad)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[:, 1:].reshape(-1),
                                                 ignore_index=VOCAB.pad_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m.build_memory(embs, ad, ce, fv)


def check_searches_constrained():
    """Every leaf of every arm satisfies the FULL formula including H, one component, <= max_len."""
    st, can, selfies, formula = nd.prep_structure("CC(=O)Oc1ccccc1C(=O)O")
    m = tiny(max_len=45)
    mem, pad = _train_one(m, can, formula)
    total = 0
    for name, kw in [("prefix_seeded", dict(l_seed=3, k_roots=3, m_rollout=8, seed=0)),
                     ("masked_beam", dict(beam=24)), ("diverse_beam", dict(beam=24, alpha=0.6))]:
        stats = nd.new_funnel()
        res = nd.SEARCHES[name](m, mem, formula, TABLE, mem_pad=pad, stats=stats, **kw)
        keys = []
        for smi, lp in res:
            mm = Chem.MolFromSmiles(smi)
            assert mm is not None, (name, smi)
            assert nd.mol_formula(mm) == formula, (name, smi, nd.mol_formula(mm))
            assert len(Chem.GetMolFrags(mm)) == 1, (name, smi)
            keys.append(nd.ik14(smi))
            total += 1
        assert len(set(keys)) == len(keys), f"{name} returned duplicate ik14"
        lps = [lp for _, lp in res]
        assert lps == sorted(lps, reverse=True), f"{name} is not sorted by logp"
        assert stats["unique"] == len(res), (name, stats)
        assert stats["eos"] >= stats["valid"] >= stats["unique"], (name, stats)
    assert total > 0, "no arm emitted a leaf, so the invariants above checked nothing"
    print(f"  all three searches: leaves match the full formula, deduped, sorted ({total} leaves)")


def check_overfit_recovery():
    """A model overfit on two molecules must be inverted by every arm: the real end-to-end check."""
    mols = []
    for s in SMILES[:2]:
        stt, can, selfies, formula = nd.prep_structure(s)
        mols.append(dict(can=can, ids=VOCAB.encode(selfies), formula=formula, ik14=nd.ik14(can)))
    maxlen = max(len(x["ids"]) for x in mols) + 4
    m = tiny(max_len=maxlen, d_model=96, n_layers=3)
    embs = torch.randn(len(mols), 1, 1024)
    ad = torch.zeros(len(mols), 1, dtype=torch.long)
    ce = torch.full((len(mols), 1), 2, dtype=torch.long)
    fv = torch.from_numpy(np.stack([nd.formula_vector(x["formula"]) for x in mols]))
    L = max(len(x["ids"]) for x in mols) + 2
    tgt = torch.full((len(mols), L), VOCAB.pad_id, dtype=torch.long)
    for i, x in enumerate(mols):
        row = [VOCAB.sos_id] + x["ids"] + [VOCAB.eos_id]
        tgt[i, :len(row)] = torch.tensor(row)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(700):
        mem, pad = m.build_memory(embs, ad, ce, fv)
        lg = m(tgt[:, :-1], mem, pad)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[:, 1:].reshape(-1),
                                                 ignore_index=VOCAB.pad_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = loss.detach().item()
    assert final < 0.05, f"the fixture did not overfit (loss {final:.4f})"
    m.eval()
    for i, x in enumerate(mols):
        mem, pad = m.build_memory(embs[i:i + 1], ad[i:i + 1], ce[i:i + 1], fv[i:i + 1])
        for name, kw in [("prefix_seeded", dict(l_seed=3, k_roots=3, m_rollout=12, seed=0)),
                         ("masked_beam", dict(beam=24)), ("diverse_beam", dict(beam=24, alpha=0.6))]:
            res = nd.SEARCHES[name](m, mem, x["formula"], TABLE, mem_pad=pad, **kw)
            keys = [nd.ik14(s) for s, _ in res]
            assert x["ik14"] in keys, f"{name} lost the memorised molecule for mol {i}: {keys[:3]}"
    print(f"  overfit recovery: every arm recovers both memorised molecules (loss {final:.4f})")


def check_morgan_and_select():
    """Our fingerprint must equal MassSpecGym's definition, and maxmin must beat logp on coverage."""
    smis = [nd.prep_structure(s)[1] for s in SMILES]
    fps, DataStructs = nd.morgan_fps(smis)
    assert all(f is not None for f in fps)
    try:                                    # the deprecated call, where it is still reachable
        from rdkit.Chem import AllChem
        ref = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smis[0]), 2, nBits=2048)
        assert list(ref.GetOnBits()) == list(fps[0].GetOnBits()), \
            "MorganGenerator disagrees with GetMorganFingerprintAsBitVect"
        legacy = "checked against the deprecated call"
    except (ImportError, AttributeError):
        legacy = "deprecated call unavailable; definition-only"
    assert nd.morgan_fps([None, ""])[0] == [None, None]

    # max_tanimoto: 1.0 against itself, 0.0 with nothing
    assert abs(nd.max_tanimoto([(smis[0], 0.0)], smis[0]) - 1.0) < 1e-9
    assert nd.max_tanimoto([], smis[0]) == 0.0

    cands = [(s, -float(i)) for i, s in enumerate(smis)]
    assert nd.select_top25(cands, "logp", k=3) == cands[:3]
    mm = nd.select_top25(cands, "maxmin", k=3)
    assert len(mm) == 3 and mm[0] == cands[0], "maxmin must be seeded by the most likely candidate"
    assert set(x[0] for x in mm) <= set(smis)
    assert len({x[0] for x in mm}) == 3
    # with near-duplicates present, maxmin must not fill up with them while logp does
    dup = [(smis[0], 0.0), (smis[0].replace("C(=O)O", "C(=O)OC"), -0.1), (smis[1], -5.0)]
    got = [s for s, _ in nd.select_top25(dup, "maxmin", k=2)]
    assert smis[1] in got, f"maxmin picked the near-duplicate over the distant candidate: {got}"
    try:
        nd.select_top25(cands, "nonsense")
        raise AssertionError("an unknown selection mode was accepted")
    except ValueError:
        pass
    print(f"  morgan_fps ({legacy}) + select_top25: OK")


def check_data_path():
    import pandas as pd
    tmp = Path(tempfile.mkdtemp())
    vp = tmp / "vocab.json"
    VOCAB.to_json(vp)
    iks = [nd.ik14(nd.prep_structure(s)[1]) for s in SMILES]
    labelf = [p3d.formula_str(nd.prep_structure(s)[3]) for s in SMILES]
    st = p3d.build_structures(iks, [nd.prep_structure(s)[1] for s in SMILES], labelf, vp, workers=1)
    assert st.in_scope.all(), st[["inchikey14", "status", "in_scope", "n_unk"]]
    assert (st.n_unk == 0).all()
    assert "scaffold_key" in st and "formula_mismatch" in st, "P1's record fields are missing"
    for _, r in st.iterrows():
        assert nd.parse_formula(r.formula) == nd.prep_structure(SMILES[iks.index(r.inchikey14)])[3]

    rows, plan = [], {0: 3, 1: 1, 2: 4, 3: 2, 4: 1, 5: 1}
    for mi, n in plan.items():
        mass = ssc.formula_mass(labelf[mi])
        for j in range(n):
            ad = nd.TEST_ADDUCTS[j % 3]
            rows.append(dict(inchikey14=iks[mi], adduct=ad, molecular_formula=labelf[mi],
                             precursor_mz=ssc.precursor_of(mass, ad)))
    rows.append(dict(inchikey14=iks[0], adduct="[M+H]+", molecular_formula=labelf[0],
                     precursor_mz=999.9))                        # contradicts the formula
    rows.append(dict(inchikey14=iks[1], adduct="[M+2H]2+", molecular_formula=labelf[1],
                     precursor_mz=100.0))                        # adduct outside the ten
    meta = pd.DataFrame(rows)
    ok = p3d.row_quality(meta)
    assert int(ok.sum()) == sum(plan.values()), (int(ok.sum()), sum(plan.values()))

    ce = pd.DataFrame({"row": [0, 1], "ce_mean": [20.0, 40.0]})
    split = pd.DataFrame({"inchikey14": iks,
                          "split": ["train", "train", "train", "dec_val", "heldout_c3", "train"]})
    idx = p3d.build_index(meta, ce, split, st, VOCAB, keep_splits=("train",))
    assert len(idx) == 4 and set(idx.ik14) == {iks[0], iks[1], iks[2], iks[5]}, idx.ik14
    for i, ik in enumerate(idx.ik14):
        mi = iks.index(ik)
        assert idx.n_spectra(i) == plan[mi], (ik, idx.n_spectra(i))
        assert list(idx.tokens(i)) == VOCAB.encode(st.set_index("inchikey14").loc[ik, "selfies"])
        for r in idx.rows(i):
            assert meta.inchikey14.iloc[r] == ik and ok[r], (ik, r)
        np.testing.assert_allclose(idx.fvec[i], nd.formula_vector(nd.parse_formula(idx.formula[i])))

    # embeddings: gather across chunks, and refuse a short middle chunk
    d = tmp / "emb_dbec3a0_lx"
    d.mkdir()
    full = np.random.default_rng(0).standard_normal((len(meta), 8), dtype=np.float32)
    CH = 5
    for c in range((len(meta) + CH - 1) // CH):
        np.save(d / f"emb_{c:04d}.npy", full[c * CH:(c + 1) * CH])
    store = p3d.EmbeddingStore(d, n_rows=len(meta), chunk=CH, dim=8)
    np.testing.assert_allclose(store.gather(range(len(meta))), full)
    bad = tmp / "emb_dbec3a0_lx_bad"
    bad.mkdir()
    np.save(bad / "emb_0000.npy", full[:3])
    np.save(bad / "emb_0001.npy", full[3:])
    try:
        p3d.EmbeddingStore(bad, chunk=CH, dim=8)
        raise AssertionError("a short middle chunk was accepted")
    except AssertionError as e:
        assert "row -> chunk mapping" in str(e), e

    # batching: padding, tgt layout, and CE dropout that never invents a CE
    rng = np.random.default_rng(1)
    b = p3d.make_batch(idx, [0, 1, 2, 3], store, rng, max_spec=2, ce_dropout=0.0,
                       pad_id=VOCAB.pad_id, sos_id=VOCAB.sos_id, eos_id=VOCAB.eos_id)
    assert b["emb"].shape[1] == 2 and (b["emb"][~b["spec_mask"]] == 0).all()
    for j, i in enumerate([0, 1, 2, 3]):
        t = list(idx.tokens(i))
        row = list(b["tgt"][j])
        assert row[0] == VOCAB.sos_id and row[1:1 + len(t)] == t
        assert row[1 + len(t)] == VOCAB.eos_id
        assert all(x == VOCAB.pad_id for x in row[2 + len(t):])
    fired = 0
    for _ in range(40):
        bb = p3d.make_batch(idx, [0, 1, 2, 3], store, rng, max_spec=4, ce_dropout=0.5,
                            pad_id=VOCAB.pad_id, sos_id=VOCAB.sos_id, eos_id=VOCAB.eos_id)
        assert not ((bb["ce"] != nd.CE_UNKNOWN) & ~bb["ce_known"] & bb["spec_mask"]).any()
        fired += int(((bb["ce"] == nd.CE_UNKNOWN) & bb["ce_known"] & bb["spec_mask"]).sum())
    assert fired > 0, "CE dropout never fired"

    order = p3d.epoch_order(len(idx), rng, 2, np.diff(idx.tok_off))
    assert sorted(int(x) for bt in order for x in bt) == list(range(len(idx)))

    ip = tmp / "index.npz"
    idx.save(ip)
    i2 = p3d.Index.load(ip)
    assert i2.ik14 == idx.ik14 and i2.formula == idx.formula and i2.meta == idx.meta
    np.testing.assert_array_equal(i2.tok_flat, idx.tok_flat)
    print("  data path: structures, quality filter, index, embeddings, batching, save/load: OK")


def check_chunkstore():
    """Resumable generation: a half-finished cohort resumes without loss or duplication."""
    tmp = Path(tempfile.mkdtemp())
    s = p3e.ChunkStore(tmp, "arm_cohort", chunk=2)
    assert s.done() == set()
    s.add("m1", [("CCO", -1.0), ("CCC", -2.0)], nd.new_funnel(), 0.1)
    s.add("m2", [], nd.new_funnel(), 0.2)          # a molecule with no candidates is still recorded
    assert s.done() == {"m1", "m2"}, s.done()
    s.add("m3", [("CCN", -3.0)], nd.new_funnel(), 0.3)
    s.flush(extra_done=["m3"])
    loaded = s.load()
    assert set(loaded) == {"m1", "m3"}, sorted(loaded)     # m2 had nothing to load
    assert loaded["m1"] == [("CCO", -1.0), ("CCC", -2.0)], loaded["m1"]

    # a fresh store over the same directory sees the done list and the parts
    s2 = p3e.ChunkStore(tmp, "arm_cohort", chunk=2)
    assert s2.done() == {"m1", "m2", "m3"}, s2.done()
    before = {p.name: p.stat().st_size for p in s2.parts()}
    s2.flush()                                     # a no-op resume must not touch the parts
    assert {p.name: p.stat().st_size for p in s2.parts()} == before
    assert s2.load() == loaded
    assert len(s2.funnels()) == 3, s2.funnels()
    # a corrupt done list degrades to "nothing done" rather than crashing the run
    (tmp / "done_arm_cohort.json").write_text("{not json")
    assert p3e.ChunkStore(tmp, "arm_cohort").done() == set()
    print("  ChunkStore: flush, resume, no duplication, corrupt done-list tolerated: OK")
    # A tag that is a prefix of another tag must not read the other's parts. `randomdenovo_X` and
    # `randomdenovo_X_pool` coexist in one directory, and a `cands_{tag}_*` glob matched both, so
    # the gated null silently loaded the pool null's candidates too (25 -> 325 per molecule) and the
    # recorded G3.1 null was not the registered one.
    with tempfile.TemporaryDirectory() as d:
        base = p3e.ChunkStore(d, "randomdenovo_pseudo_c3", chunk=1)
        pool = p3e.ChunkStore(d, "randomdenovo_pseudo_c3_pool", chunk=1)
        base.add("m1", [("CCO", 0.0)], dict(sequences=1, unique=1), 0.1)
        pool.add("m1", [("CCO", 0.0), ("CCC", 0.0), ("CCN", 0.0)], dict(sequences=3, unique=3), 0.1)
        base.flush(); pool.flush()
        assert len(base.load()["m1"]) == 1, base.load()["m1"]
        assert len(pool.load()["m1"]) == 3, pool.load()["m1"]
        assert len(base.parts()) == 1 and len(pool.parts()) == 1
    print("  ChunkStore: a prefix tag does not read a longer tag's parts: OK")



def check_eval_helpers():
    can = nd.prep_structure(SMILES[0])[1]
    q = dict(molecule_id="m1", ik14=nd.ik14(can), can=can, formula=p3e.oracle_formula(can),
             src_lib="gnps", timstof=False, n_ce_known=1)
    assert nd.parse_formula(q["formula"]) == nd.prep_structure(SMILES[0])[3]
    # a candidate list containing the truth scores exact@1 and Tanimoto 1.0, in both modes
    sc = p3e.score_molecule(q, [(can, 0.0), (nd.prep_structure(SMILES[1])[1], -1.0)])
    for mode in p3e.SELECT_MODES:
        assert abs(sc[f"maxtan_{mode}"] - 1.0) < 1e-9, (mode, sc)
        assert sc[f"exact1_{mode}"] == 1 and sc[f"exact25_{mode}"] == 1, (mode, sc)
    miss = p3e.score_molecule(q, [(nd.prep_structure(SMILES[1])[1], 0.0)])
    assert miss["exact25_maxmin"] == 0 and miss["maxtan_maxmin"] < 1.0
    empty = p3e.score_molecule(q, [])
    assert empty["n_cands"] == 0 and empty["maxtan_maxmin"] == 0.0

    summ = p3e.summarise([sc, miss, empty], "x")
    assert summ["n"] == 3 and abs(summ["zero_cand_share"] - 1 / 3) < 1e-9

    # the winner rule: strictly greater wins, ties fall to A then C then B
    bake = {"prefix_seeded": {"median_maxtan_maxmin": 0.2},
            "masked_beam": {"median_maxtan_maxmin": 0.3},
            "diverse_beam": {"median_maxtan_maxmin": 0.3}}
    assert p3e.pick_winner(bake)[0] == "diverse_beam", p3e.pick_winner(bake)
    bake["prefix_seeded"]["median_maxtan_maxmin"] = 0.3
    assert p3e.pick_winner(bake)[0] == "prefix_seeded", "a tie must fall to arm A"

    # G3: the threshold is on the DELTA, and G3.2 can veto a passing G3.1
    v = p3e.g3_verdict({"median_maxtan_maxmin": 0.30}, {"median_maxtan_maxmin": 0.20}, 0.10, True)
    assert v["verdict"] == "PASS" and abs(v["delta"] - 0.10) < 1e-12, v
    v = p3e.g3_verdict({"median_maxtan_maxmin": 0.29}, {"median_maxtan_maxmin": 0.20}, 0.10, True)
    assert v["verdict"] == "FAIL" and v["items"]["G3.1"] == "FAIL", v
    v = p3e.g3_verdict({"median_maxtan_maxmin": 0.40}, {"median_maxtan_maxmin": 0.20}, 0.10, False)
    assert v["verdict"] == "FAIL" and v["items"]["G3.1"] == "PASS" and v["items"]["G3.2"] == "FAIL", v
    print("  eval helpers: scoring symmetry, summaries, winner rule, G3 verdict: OK")


def check_randomdenovo_unpacking():
    """The null arm must survive `generate_random_molecule_graphs_via_traversal` returning a LIST.

    It is typed `-> list[Mol]` and yields up to `max_top_k` molecules per call. An earlier version of
    the notebook passed that list straight to `Chem.MolToSmiles`, which raises `ArgumentError`; the
    worker's `except Exception` turned every raise into "0 candidates", so all 755 molecules of the
    null arm came back empty and the G3.1 delta was just the treatment's own score. A null of exactly
    zero is a code fault, never a measurement."""
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1", "CC(=O)O")]

    class Gen:
        def __init__(self, ret):
            self.ret, self.calls = ret, 0

        def generate_random_molecule_graphs_via_traversal(self, formula):
            self.calls += 1
            return self.ret

    # the real shape: a list of Mols
    got = p3e.randomdenovo_smiles(Gen(mols), "C2H6O")
    assert got == [Chem.MolToSmiles(m) for m in mols], got

    # tolerated shapes: a bare Mol, a SMILES string, None, an empty list, a list holding a None
    assert p3e.randomdenovo_smiles(Gen(mols[0]), "C2H6O") == ["CCO"]
    assert p3e.randomdenovo_smiles(Gen(["CCO"]), "C2H6O") == ["CCO"]
    assert p3e.randomdenovo_smiles(Gen(None), "C2H6O") == []
    assert p3e.randomdenovo_smiles(Gen([]), "C2H6O") == []
    assert p3e.randomdenovo_smiles(Gen([mols[0], None, mols[1]]), "X") == ["CCO", "c1ccccc1"]

    # one call per invocation: the caller controls the draw count, not the helper
    g = Gen(mols)
    p3e.randomdenovo_smiles(g, "C2H6O")
    assert g.calls == 1, g.calls

    # and the notebook must not have grown a second, unguarded unpacking
    src = (HERE / "novel_p3_null_colab.py").read_text(encoding="utf-8")
    assert "p3e.randomdenovo_smiles" in src, "the null notebook stopped using the shared helper"
    assert "MolToSmiles(g)" not in src and "MolToSmiles(got)" not in src, (
        "the null notebook converts a generator return value to SMILES itself again; that is the "
        "bug that zeroed the whole null arm")
    # an all-zero null must be refused, not scored
    assert "cannot be judged against an empty null" in src, (
        "the guard against scoring an empty null arm is gone")
    print("  RandomDeNovo return-value unpacking + the empty-null guard: OK")


def check_checkpoint_roundtrip():
    """Save/load must restore the optimizer AND the RNG, so a resumed run continues identically."""
    tmp = Path(tempfile.mkdtemp())
    m = tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    fv = torch.from_numpy(nd.formula_vector({"C": 6, "H": 6}))[None].repeat(2, 1)
    embs, ad, ce = torch.randn(2, 1, 1024), torch.zeros(2, 1, dtype=torch.long), torch.full((2, 1), 5)
    tgt = torch.full((2, 6), VOCAB.pad_id, dtype=torch.long)
    tgt[:, 0] = VOCAB.sos_id
    tgt[:, 1] = VOCAB.tok2id["[C]"]
    tgt[:, 2] = VOCAB.eos_id

    def step():
        mem, pad = m.build_memory(embs, ad, ce, fv)
        lg = m(tgt[:, :-1], mem, pad)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[:, 1:].reshape(-1),
                                                 ignore_index=VOCAB.pad_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.detach().item()

    for _ in range(3):
        step()
    rng = np.random.default_rng(11)
    rng.random(5)
    ck = dict(model=m.state_dict(), opt=opt.state_dict(),
              rng=dict(torch=torch.get_rng_state(), numpy=rng.bit_generator.state))
    dest = tmp / "c.pt"
    tmp_path = Path(str(dest) + ".tmp")
    torch.save(ck, tmp_path)
    os.replace(tmp_path, dest)                     # the notebook's atomic-save shape
    want = [step() for _ in range(3)]
    want_rand = rng.random(4).tolist()

    m2, opt2 = tiny(), None
    loaded = torch.load(dest, map_location="cpu", weights_only=False)
    m2.load_state_dict(loaded["model"])
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(loaded["opt"])
    torch.set_rng_state(loaded["rng"]["torch"])
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = loaded["rng"]["numpy"]
    m, opt = m2, opt2
    got = [step() for _ in range(3)]
    assert np.allclose(want, got, atol=1e-6), f"resumed training diverged: {want} vs {got}"
    assert np.allclose(want_rand, rng2.random(4).tolist()), "the numpy RNG state did not restore"

    # a truncated file must fail to load, so the notebook can fall back to the previous checkpoint
    dest.write_bytes(b"garbage")
    try:
        torch.load(dest, map_location="cpu", weights_only=False)
        raise AssertionError("a truncated checkpoint loaded")
    except AssertionError:
        raise
    except Exception:
        pass
    print("  checkpoint round-trip (model, optimizer, torch + numpy RNG) and truncation: OK")


def check_notebooks():
    """Generated notebooks are fresh, parse, and embed build/ sources verbatim."""
    import mk_novel_p0 as mk
    p3 = [b for b in mk.BUILDS if "p3" in b[0].name]
    assert len(p3) == 3, [b[0].name for b in p3]
    for src, dst, tags in mk.BUILDS:
        before = dst.read_text(encoding="utf-8") if dst.exists() else None
        mk.build(src, dst, tags)
        assert before is None or before == dst.read_text(encoding="utf-8"), \
            f"{dst.name} was stale -- it has now been regenerated; re-run the test"
        nb = json.loads(dst.read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i, c in enumerate(code):
            ast.parse(c, filename=f"{dst.name}[code {i}]")
        for tag in tags:
            var, path = mk.SOURCES[tag]
            lines = [ln for c in code for ln in c.split("\n") if ln.startswith(f"{var} = ")]
            assert len(lines) == 1, (dst.name, var, len(lines))
            assert ast.literal_eval(lines[0][len(var) + 3:]) == path.read_text(encoding="utf-8"), \
                f"{dst.name}: {var} is not {path.name} verbatim"
    # the pre-registered G3 numbers must match what the notebooks actually run
    gr = (ROOT / "specs_novel" / "gate_report.md").read_text(encoding="utf-8")
    ev = (HERE / "novel_p3_eval_colab.py").read_text(encoding="utf-8")
    tr = (HERE / "novel_p3_train_colab.py").read_text(encoding="utf-8")
    assert "## G3 — Decoder training and constrained search (P3)" in gr, "G3 is not pre-registered"
    for token in ("N_CANDIDATES = 256", "L_SEED, K_ROOTS, M_ROLLOUT = 4, 8, 32", "BEAM = 256",
                  "ALPHA_DIVERSE = 0.6", "DEC_VAL_SUBSET = 300"):
        assert token in ev, f"eval notebook lost {token}"
    for token in ("D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 512, 8, 8, 2048, 0.1",
                  "MAX_SPEC_PER_MOL = 4", "CE_DROPOUT = 0.3", "MAX_EPOCHS, PATIENCE = 100, 5"):
        assert token in tr, f"train notebook lost {token}"
    print(f"  notebooks: fresh, parse, embed sources verbatim; G3 constants match ({len(mk.BUILDS)})")


if __name__ == "__main__":
    check_constants()
    check_memory()
    check_tied_and_forward()
    check_searches_constrained()
    check_overfit_recovery()
    check_morgan_and_select()
    check_data_path()
    check_chunkstore()
    check_eval_helpers()
    check_randomdenovo_unpacking()
    check_checkpoint_roundtrip()
    check_notebooks()
    print("test_novel_p3: OK")
