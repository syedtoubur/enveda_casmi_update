"""Run-mode detection and submission writing."""
import os

import pandas as pd


def run_mode(n_mol, n_spec, cfg, env=None):
    """-> (fast, info). Fail-safe: full run unless the PUBLIC test file is recognised and this is not the rerun."""
    env = os.environ if env is None else env
    raw = env.get("KAGGLE_IS_COMPETITION_RERUN")
    looks_public = n_mol == cfg.public_n_mol and n_spec == cfg.public_n_spec
    fast = bool(cfg.allow_fast) and looks_public and not bool(raw) and not cfg.force_full
    return fast, dict(rerun_env=raw, looks_public=looks_public, mode="fast" if fast else "full")


def make_submission(ranked, sample, topn=25, pad="CCO"):
    """{molecule_id: [smiles]} aligned to sample_submission, exactly topn per row, audited."""
    sub = sample[["molecule_id"]].copy()
    sub["smiles"] = [";".join((list(ranked.get(m, [])) + [pad] * topn)[:topn]) for m in sub.molecule_id]
    assert len(sub) == len(sample) and (sub.molecule_id.values == sample.molecule_id.values).all()
    assert not sub.molecule_id.duplicated().any() and sub.smiles.notna().all()
    assert (sub.smiles.str.split(";").map(len) == topn).all()
    return sub


def write_submission(sub, path="submission.csv"):
    sub.to_csv(path, index=False)
    return path


def molecules(test_df):
    return [(m, g) for m, g in test_df.groupby("molecule_id")]


def read_sample(path):
    return pd.read_csv(path)
