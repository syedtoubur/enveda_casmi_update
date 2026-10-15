"""CPU dry run of casmi_novel_p4_colab.ipynb against a fake Drive (no Colab, no GPU).

Reuses novel_p3_dryrun: fake Drive -> 2-epoch tiny training (writes p3/decoder.pt) -> a synthetic p4_inputs.pkl
(per cohort molecule: its oracle formula plus a wrong one; mutants = the truth and a decoy) -> the P4 notebook, twice
(the second run must resume with nothing to do) -> p4_raw.pkl checks.

    python build/novel_p4_dryrun.py
"""
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import novel_p3_dryrun as d3  # noqa: E402
import novel_p3_eval as p3e  # noqa: E402

import pandas as pd  # noqa: E402


def p4_subs(drive_dir, chunk):
    return [
        ('from google.colab import drive\ndrive.mount("/content/drive")', 'drive = None  # dry run'),
        ('ROOT = Path("/content/drive/MyDrive/casmi_2026")', f'ROOT = Path({str(drive_dir)!r})'),
        ('SRC = Path("/content/novel_p4/src")', f'SRC = Path({str(Path(drive_dir).parent / "p4src")!r})'),
        ('assert v_match(rdkit.__version__, RDKIT_PIN), rdkit.__version__\n'
         'assert torch.cuda.is_available(), "P4 needs a GPU runtime"\n'
         'DEV = torch.device("cuda")',
         'DEV = torch.device("cpu")'),
        ('gpu=torch.cuda.get_device_name(0)', 'gpu="dry-run-cpu"'),
        ('sh(f"{sys.executable} -m pip install -q selfies=={SELFIES_PIN}", check=False)', 'pass'),
        ('N_CORPUS_ROWS = 1_323_878', 'N_CORPUS_ROWS = None'),
        ('STORE = p3d.EmbeddingStore(EMB, n_rows=N_CORPUS_ROWS)', f'STORE = p3d.EmbeddingStore(EMB, chunk={chunk})'),
        ('BEAM = 64 ', 'BEAM = 4 '),
        ('SUB256 = 150 ', 'SUB256 = 2 '),
        ('CHUNK = 25 ', 'CHUNK = 2 '),
        ('beam=256, max_len=MAX_LEN, alpha=0.0)]', 'beam=6, max_len=MAX_LEN, alpha=0.0)]'),
    ]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="novel_p4_dry_"))
    drive_dir = tmp / "casmi_2026"
    try:
        facts = d3.build_fake_drive(drive_dir)
        text = d3.patch(d3.nb_code(ROOT / "casmi_novel_p3_train_colab.ipynb"),
                        d3.train_subs(drive_dir, facts, facts["chunk"]))
        d3.run(text.replace("MAX_EPOCHS, PATIENCE = 100, 5", "MAX_EPOCHS, PATIENCE = 2, 99"), "train")
        assert (drive_dir / "_novel" / "p3" / "decoder.pt").exists()
        q, _ = p3e.build_queries(pd.read_parquet(drive_dir / "_novel" / "p1" / "pseudo_c3.parquet"))
        rows = {}
        for x in q:
            wrong = x["formula"].replace("C", "C1", 1) if "C" in x["formula"] else x["formula"]
            rows[x["molecule_id"]] = dict(formulas=[x["formula"], wrong], has_formula=True, acyclic=False,
                                          timstof=False, mutants=[(x["can"], x["formula"], 0, "+CH2"),
                                                                  ("CCO", "C2H6O", 0, "+O")])
        out = drive_dir / "_novel" / "p4"
        out.mkdir(parents=True, exist_ok=True)
        pickle.dump(dict(rows=rows, cfg={}), open(out / "p4_inputs.pkl", "wb"))
        code = d3.nb_code(ROOT / "casmi_novel_p4_colab.ipynb")
        ptext = d3.patch(code, p4_subs(drive_dir, facts["chunk"]))
        ns = d3.run(ptext, "p4")
        raw = pickle.load(open(out / "p4_raw.pkl", "rb"))["raw"]
        assert set(raw) == set(rows), (len(raw), len(rows))
        n_dec = sum(len(r["dec"]) for r in raw.values())
        n_mut = sum(r["mut"][0][1] is not None for r in raw.values() if r["mut"])
        assert all(len(r["mut"]) == 2 for r in raw.values())
        assert sum(bool(r["dec256"]) for r in raw.values()) <= 2
        print(f"P4 dry run OK: {len(raw)} molecules, {n_dec} decoder leaves, truth teacher-scored for {n_mut}")
        ns2 = d3.run(ptext, "p4-resume")
        assert not ns2["todo"], "resume re-ran finished molecules"
        print("resume OK: nothing re-run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
