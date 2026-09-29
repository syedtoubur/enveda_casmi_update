"""Dry-run the P3 notebooks locally: a synthetic Drive, a stubbed `google.colab`, a tiny model.

    python build/novel_p3_dryrun.py            # train notebook, fresh + kill-and-resume rehearsal
    python build/novel_p3_dryrun.py --keep     # keep the fake Drive for inspection

Why this exists: the P1 notebook was validated this way and it is what caught its problems before a
Colab hour was spent. The substitutions below are all **environment** -- device, version pins,
corpus size, model size -- and never the logic under test: the structure pass, the index build, the
exclusion assertions, the training loop, and the checkpoint / resume path all run for real.

Every substitution asserts it matched exactly once, so if the notebook source drifts this harness
fails loudly instead of quietly testing nothing.
"""
import argparse
import ast
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

N_MOL = 48
N_ROWS = 140
SEED = 7

# molecules whose neutral mass sits inside novel_cohort.label_ok's [157, 1159] window -- anything
# lighter is filtered out as a bad label and the index would come back empty
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",
    "CCCCCCCCCCCCCCCC(=O)O", "c1ccc2c(c1)ccc1ccccc12", "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OCC1OC(O)C(O)C(O)C1O", "Cc1ccc2c(c1)[nH]c1c2CCNC1",
    "O=C(O)c1ccccc1Nc1ccccc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", "COc1cc(C=CC(=O)O)ccc1O", "Oc1ccc(CCN)cc1O",
    "CC1=CC(=O)CC(C)(C)C1", "NC(Cc1ccc(O)cc1)C(=O)O", "OCC(O)C(O)C(O)C(O)CO",
    "Cc1cc(C)cc(OCC(=O)O)c1", "O=C(Nc1ccccc1)c1ccccc1",
]


def build_fake_drive(d):
    """A miniature but structurally faithful Drive: corpus, CE, vocab, split, embeddings."""
    import selfies as sf
    import novel_decoder as nd
    import novel_p3_data as p3d
    import simsim_common as ssc

    rng = np.random.default_rng(SEED)
    d = Path(d)
    (d / "_dreams_out").mkdir(parents=True, exist_ok=True)
    (d / "_novel" / "p0").mkdir(parents=True, exist_ok=True)
    (d / "_novel" / "p1").mkdir(parents=True, exist_ok=True)
    emb_dir = d / "_novel" / "emb_dbec3a0_lx"
    emb_dir.mkdir(parents=True, exist_ok=True)
    (d / "casmi26-offline-rdkit-2026033").mkdir(parents=True, exist_ok=True)

    smi = [SMILES[i % len(SMILES)] for i in range(N_MOL)]
    iks, formulas, keep, alpha = [], [], [], set()
    for s in smi:
        st, can, selfies, counts = nd.prep_structure(s)
        assert st == "ok", (s, st)
        k = nd.ik14(can)
        if k in iks:
            continue
        iks.append(k)
        formulas.append(p3d.formula_str(counts))
        keep.append(can)
        alpha |= set(sf.split_selfies(selfies))
    n_mol = len(iks)

    # the vocabulary is the corpus alphabet UNION the robust alphabet -- exactly how P0.3 built the
    # real one. Building it from the robust alphabet alone leaves <unk> tokens, and the notebook's
    # own assertion catches that (as it should).
    vocab = nd.SelfiesVocab.from_alphabet(alpha | set(sf.get_semantic_robust_alphabet()))
    vocab.to_json(d / "_novel" / "p0" / "selfies_vocab.json")

    rows = []
    for i in range(N_ROWS):
        mi = i % n_mol
        mass = ssc.formula_mass(formulas[mi])
        ad = nd.TEST_ADDUCTS[i % len(nd.TEST_ADDUCTS)]
        rows.append(dict(ingest_lib=["gnps", "mona", "riken"][i % 3], inchikey14=iks[mi],
                         normalized_smiles=keep[mi], molecular_formula=formulas[mi],
                         ionization_mode="positive" if "+" in ad else "negative", adduct=ad,
                         precursor_mz=ssc.precursor_of(mass, ad), oob_mass=False))
    meta = pd.DataFrame(rows)
    meta.to_parquet(d / "_dreams_out" / "corpus_meta.parquet", index=False)

    n_ce = rng.integers(0, 3, len(meta))
    ce = pd.DataFrame({"row": np.arange(len(meta), dtype=np.int64),
                       "train_row": np.arange(len(meta), dtype=np.int64),
                       "n_ce": n_ce,
                       "ce_mean": np.where(n_ce > 0, rng.uniform(10, 60, len(meta)), np.nan),
                       "instrument_type": "QTOF"})
    ce.to_parquet(d / "_novel" / "p0" / "corpus_ce.parquet", index=False)

    # split: most train, a few dec_val, a few held out -- the held-out ones must never be trained on
    labels = []
    for i in range(n_mol):
        labels.append("dec_val" if i % 7 == 3 else ("heldout_c3" if i % 11 == 5 else "train"))
    pd.DataFrame({"inchikey14": iks, "scaffold_key": [f"s{i}" for i in range(n_mol)],
                  "split": labels}).to_parquet(d / "_novel" / "p1" / "dec_split.parquet", index=False)

    chunk = 64
    full = rng.standard_normal((len(meta), 1024)).astype(np.float32)
    for c in range((len(meta) + chunk - 1) // chunk):
        np.save(emb_dir / f"emb_{c:04d}.npy", full[c * chunk:(c + 1) * chunk])

    # cohort parquets for the eval leg: the held-out molecules stand in for pseudo_c3 / ts_dec, with
    # the same columns the P1 notebook writes
    by_ik = {k: g for k, g in meta.groupby("inchikey14")}

    def cohort(ik_list, timstof):
        rows = []
        for ik in ik_list:
            g = by_ik[ik]
            mi = iks.index(ik)
            rows.append(dict(molecule_id=f"m_{ik}", inchikey14=ik, smiles=keep[mi], can=keep[mi],
                             selfies=nd.prep_structure(keep[mi])[2], formula=formulas[mi],
                             scaffold=f"s{mi}", group_size=1,
                             src_lib="enveda-np-examples" if timstof else "gnps",
                             timstof=timstof, in_scope=True,
                             rows=np.asarray(g.index.to_numpy()[:4], np.int64),
                             train_rows=np.zeros(0, np.int64),
                             adducts=[str(a) for a in g.adduct.to_numpy()[:4]],
                             ces=[20.0, None, 40.0, None][:len(g.index[:4])],
                             lib_ban_keys=[ik], pool_ban_keys=[ik], dreams_ft_risk=False))
        return pd.DataFrame(rows)

    heldout = [ik for ik, lab in zip(iks, labels) if lab == "heldout_c3"]
    trainish = [ik for ik, lab in zip(iks, labels) if lab == "train"][:3]
    cohort(heldout, False).to_parquet(d / "_novel" / "p1" / "pseudo_c3.parquet", index=False)
    cohort(trainish, True).to_parquet(d / "_novel" / "p1" / "ts_dec.parquet", index=False)

    return dict(n_rows=len(meta), n_ik14=n_mol, chunk=chunk,
                labels=pd.Series(labels).value_counts().to_dict(),
                n_c3=len(heldout), n_ts=len(trainish))


# --------------------------------------------------------------------------------------------- #
# Notebook rewriting: environment only
# --------------------------------------------------------------------------------------------- #

def nb_code(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def patch(code, subs):
    text = "\n\n# %%\n".join(code)
    for old, new in subs:
        assert text.count(old) == 1, f"dry-run substitution matched {text.count(old)}x: {old!r}"
        text = text.replace(old, new)
    return text


def train_subs(drive_dir, facts, chunk):
    return [
        # -- environment: no Colab, no Drive, no GPU --
        ('from google.colab import drive\ndrive.mount("/content/drive")',
         'drive = None  # dry run'),
        ('ROOT = Path("/content/drive/MyDrive/casmi_2026")',
         f'ROOT = Path({str(drive_dir)!r})'),
        ('WORK = Path("/content/novel_p3")', f'WORK = Path({str(Path(drive_dir).parent / "work")!r})'),
        ('assert torch.cuda.is_available(), "P3 training needs a GPU runtime"\n'
         'DEV = torch.device("cuda")\nGPU = torch.cuda.get_device_name(0)\n'
         'BF16 = torch.cuda.is_bf16_supported()',
         'DEV = torch.device("cpu")\nGPU = "dry-run-cpu"\nBF16 = False'),
        # -- version pins: the local kernel is not Colab's --
        ('assert v_match(rdkit.__version__, RDKIT_PIN), rdkit.__version__\n'
         'assert dist_version("selfies") == SELFIES_PIN, dist_version("selfies")',
         'print("dry run: version pins not enforced", rdkit.__version__, dist_version("selfies"))'),
        ('assert vsha.startswith(VOCAB_SHA1), f"selfies_vocab.json sha1 {vsha} != {VOCAB_SHA1}..."',
         'print("dry run: vocab sha1 not enforced", vsha)'),
        # -- corpus size --
        (f'N_CORPUS_ROWS = 1_323_878                        # P0.2\nN_CORPUS_IK14 = 194_668',
         f'N_CORPUS_ROWS = {facts["n_rows"]}\nN_CORPUS_IK14 = {facts["n_ik14"]}'),
        ('STORE = p3d.EmbeddingStore(EMB, n_rows=N_CORPUS_ROWS)',
         f'STORE = p3d.EmbeddingStore(EMB, n_rows=N_CORPUS_ROWS, chunk={chunk})'),
        # -- a tiny model, so the loop runs in seconds --
        ('D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 512, 8, 8, 2048, 0.1',
         'D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 32, 2, 2, 64, 0.0'),
        ('BATCH = 256                                      # molecules per step',
         'BATCH = 8'),
        ('LR, WARMUP, COSINE_EPOCHS, LR_MIN_FRAC = 3e-4, 1000, 40, 0.1',
         'LR, WARMUP, COSINE_EPOCHS, LR_MIN_FRAC = 3e-3, 5, 8, 0.1'),
        ('VAL_BATCHES = 24', 'VAL_BATCHES = 2'),
        # -- CPU autocast would need float32; just disable it --
        ('with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):\n'
         '            loss, nt, nc = batch_loss(d)',
         'if True:\n            loss, nt, nc = batch_loss(d)'),
        ('with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):\n'
         '            loss, _, _ = batch_loss(b)',
         'if True:\n            loss, _, _ = batch_loss(b)'),
        ('with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):\n'
         '            loss, nt, nc = batch_loss(b)',
         'if True:\n            loss, nt, nc = batch_loss(b)'),
        ('        if i == CAL_WARMUP - 1:                   # start the clock after the warmup steps\n'
         '            torch.cuda.synchronize()\n            t0 = time.time()\n'
         '    torch.cuda.synchronize()',
         '        if i == CAL_WARMUP - 1:\n            t0 = time.time()\n    pass'),
        ('                cuda=torch.cuda.get_rng_state_all(),', '                cuda=None,'),
    ]


def eval_subs(drive_dir, chunk):
    """Environment-only substitutions for the eval notebook: no Colab, CPU, small budgets."""
    return [
        ('from google.colab import drive\ndrive.mount("/content/drive")', 'drive = None  # dry run'),
        ('ROOT = Path("/content/drive/MyDrive/casmi_2026")', f'ROOT = Path({str(drive_dir)!r})'),
        ('WORK = Path("/content/novel_p3_eval")',
         f'WORK = Path({str(Path(drive_dir).parent / "work_eval")!r})'),
        ('assert torch.cuda.is_available(), "the bake-off needs a GPU runtime"\n'
         'DEV = torch.device("cuda")',
         'DEV = torch.device("cpu")'),
        ('gpu=torch.cuda.get_device_name(0),', 'gpu="dry-run-cpu",'),
        ('assert v_match(rdkit.__version__, RDKIT_PIN), rdkit.__version__\n'
         'assert dist_version("selfies") == SELFIES_PIN, dist_version("selfies")',
         'print("dry run: version pins not enforced")'),
        ('N_CORPUS_ROWS = 1_323_878', f'N_CORPUS_ROWS = None'),
        ('STORE = p3d.EmbeddingStore(EMB, n_rows=N_CORPUS_ROWS)',
         f'STORE = p3d.EmbeddingStore(EMB, chunk={chunk})'),
        # budgets: the search itself still runs for real, just narrower
        ('L_SEED, K_ROOTS, M_ROLLOUT = 4, 8, 32             # 8 x 32 = 256',
         'L_SEED, K_ROOTS, M_ROLLOUT = 3, 2, 4'),
        ('BEAM = 256', 'BEAM = 8'),
        ('DEC_VAL_SUBSET = 300', 'DEC_VAL_SUBSET = 3'),
        ('CHUNK_MOL = 50                                    # flush to Drive every 50 molecules',
         'CHUNK_MOL = 2'),
        ('SAMPLE_UNCONSTRAINED = 64                         # samples per molecule for the no-mask diagnostic\n'
         'UNCONSTRAINED_MOLS = 100',
         'SAMPLE_UNCONSTRAINED = 4\nUNCONSTRAINED_MOLS = 2'),
        ('tf_loss, tf_acc = teacher_forced(VAL_IDX)', 'tf_loss, tf_acc = teacher_forced(VAL_IDX, 1, 4)'),
    ]


def run(text, tag, extra=None):
    ns = {"__name__": "__main__", "__file__": f"<dryrun {tag}>"}
    ns.update(extra or {})
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google = types.ModuleType("google")
    google.colab = colab
    sys.modules.setdefault("google", google)
    sys.modules["google.colab"] = colab
    ast.parse(text)                                  # syntax first, so a typo is not a runtime error
    exec(compile(text, f"<dryrun {tag}>", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--nb", default=str(ROOT / "casmi_novel_p3_train_colab.ipynb"))
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="novel_p3_dry_"))
    drive_dir = tmp / "casmi_2026"
    try:
        facts = build_fake_drive(drive_dir)
        print(f"fake Drive at {drive_dir}: {facts}")
        code = nb_code(args.nb)
        print(f"notebook {Path(args.nb).name}: {len(code)} code cells")

        # ---- leg 1: a fresh run to early stop ----
        text = patch(code, train_subs(drive_dir, facts, facts["chunk"]))
        text_fresh = text.replace("MAX_EPOCHS, PATIENCE = 100, 5", "MAX_EPOCHS, PATIENCE = 6, 99")
        assert "MAX_EPOCHS, PATIENCE = 6, 99" in text_fresh
        print("\n" + "=" * 88 + "\nLEG 1: fresh run, 6 epochs\n" + "=" * 88)
        ns = run(text_fresh, "train-fresh")
        hist1 = [dict(r) for r in ns["HIST"]]
        assert len(hist1) == 6, len(hist1)
        assert ns["START"] == 0, "leg 1 should have started fresh"
        p3 = drive_dir / "_novel" / "p3"
        assert (p3 / "decoder.pt").exists() and (p3 / "train_log.jsonl").exists()
        cks = sorted((p3 / "ckpt").glob("epoch_*.pt"))
        assert len(cks) == 2, [c.name for c in cks]       # rotation keeps two
        assert cks[-1].name == "epoch_005.pt", cks[-1].name
        print(f"leg 1 OK: 6 epochs, dec_val {hist1[0]['dec_val_loss']:.4f} -> "
              f"{hist1[-1]['dec_val_loss']:.4f}, {len(cks)} checkpoints kept")

        # ---- leg 2: kill after epoch 2, resume, and land on the same numbers ----
        print("\n" + "=" * 88 + "\nLEG 2: kill after epoch 2, then resume to 6\n" + "=" * 88)
        shutil.rmtree(p3)
        text_a = text.replace("MAX_EPOCHS, PATIENCE = 100, 5", "MAX_EPOCHS, PATIENCE = 3, 99")
        ns_a = run(text_a, "train-kill")
        assert len(ns_a["HIST"]) == 3, len(ns_a["HIST"])
        print(f"  killed after epoch {ns_a['HIST'][-1]['epoch']}")
        ns_b = run(text_fresh, "train-resume")
        assert ns_b["START"] == 3, f"resume started at {ns_b['START']}, expected 3"
        hist2 = [dict(r) for r in ns_b["HIST"]]
        assert len(hist2) == 6, len(hist2)

        # the resumed run must reproduce the uninterrupted one: this is what the RNG state buys
        bad = [(a["epoch"], a["dec_val_loss"], b["dec_val_loss"])
               for a, b in zip(hist1, hist2) if abs(a["dec_val_loss"] - b["dec_val_loss"]) > 1e-6]
        for e, x, y in bad:
            print(f"  epoch {e}: fresh {x:.8f} vs resumed {y:.8f}")
        assert not bad, "a resumed run diverged from the uninterrupted one"
        print(f"leg 2 OK: resumed at epoch 3 and reproduced all 6 epochs of leg 1 exactly")

        # ---- leg 3: a truncated newest checkpoint must fall back, not crash ----
        print("\n" + "=" * 88 + "\nLEG 3: truncated newest checkpoint\n" + "=" * 88)
        cks = sorted((p3 / "ckpt").glob("epoch_*.pt"))
        cks[-1].write_bytes(b"not a torch file")
        ns_c = run(text.replace("MAX_EPOCHS, PATIENCE = 100, 5", "MAX_EPOCHS, PATIENCE = 7, 99"),
                   "train-truncated")
        assert ns_c["START"] == int(cks[-2].stem.split("_")[1]) + 1, ns_c["START"]
        print(f"leg 3 OK: fell back to {cks[-2].name} and resumed at epoch {ns_c['START']}")

        # ---- leg 4: the eval notebook, then a resume of it ----
        print("\n" + "=" * 88 + "\nLEG 4: eval notebook (bake-off) + generation resume\n" + "=" * 88)
        nb_eval = ROOT / "casmi_novel_p3_eval_colab.ipynb"
        etext = patch(nb_code(nb_eval), eval_subs(drive_dir, facts["chunk"]))
        ns_e = run(etext, "eval")
        assert ns_e["WINNER"] in ns_e["ARMS"], ns_e["WINNER"]
        assert len(ns_e["BAKE"]) == 3, sorted(ns_e["BAKE"])
        cand_dir = drive_dir / "_novel" / "p3" / "cands"
        parts = sorted(p.name for p in cand_dir.glob("cands_*.parquet"))
        assert parts, "no candidate parquet was written"
        assert (drive_dir / "_novel" / "p3" / "p3_eval.json").exists()
        ev = json.loads((drive_dir / "_novel" / "p3" / "p3_eval.json").read_text())
        for key in ("bake_off", "winner", "cohorts", "teacher_forced", "unconstrained", "queries"):
            assert key in ev, key
        print(f"leg 4 OK: winner {ns_e['WINNER']}, {len(parts)} candidate parts, "
              f"cohorts {sorted(ev['cohorts'])}")

        # every generated leaf must satisfy its molecule's FULL formula -- the gate rests on this
        import novel_decoder as _nd
        from rdkit import Chem
        checked = 0
        for label in ("pseudo_c3", "ts_dec"):
            q = {x["molecule_id"]: x for x in (ns_e["Q_C3"] if label == "pseudo_c3" else ns_e["Q_TS"])}
            df = pd.concat([pd.read_parquet(p) for p in
                            cand_dir.glob(f"cands_{ns_e['WINNER']}_{label}_*.parquet")],
                           ignore_index=True)
            for _, r in df[df["rank"] >= 0].iterrows():
                m = Chem.MolFromSmiles(r["smiles"])
                want = _nd.parse_formula(q[r["molecule_id"]]["formula"])
                assert m is not None and _nd.mol_formula(m) == want, (label, r["smiles"])
                assert len(Chem.GetMolFrags(m)) == 1, (label, r["smiles"])
                checked += 1
        print(f"leg 4 OK: all {checked} written leaves match their full formula, single component")

        # a resumed eval run must add nothing and re-derive the same winner
        before = {p.name: p.stat().st_size for p in cand_dir.glob("cands_*.parquet")}
        ns_e2 = run(etext, "eval-resume")
        after = {p.name: p.stat().st_size for p in cand_dir.glob("cands_*.parquet")}
        assert before == after, f"a resumed eval run rewrote candidates: {before} -> {after}"
        assert ns_e2["WINNER"] == ns_e["WINNER"], (ns_e2["WINNER"], ns_e["WINNER"])
        print("leg 4 OK: a resumed eval run generated nothing new and picked the same winner")

        print("\n" + "=" * 88)
        print("DRYRUN: OK")
        print("=" * 88)
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
