"""Merge Stage-C shards into the final rank_train.npz matrix.

Follows the exact contract and assertions defined in RUNBOOK.md and casmi_rank_matrix.ipynb:
- X.shape[1] == 31 (exact 31 feature columns)
- np.isfinite(X).all() (no NaNs or Infs)
- 0.35 < (M == 0).mean() < 0.65 (balanced Class-1 / Class-2 split)
- Compressed numpy archive with keys: 'X', 'Y', 'M'
"""
import os
import sys
import numpy as np

def load_shard(path):
    """Load shard whether it is a .npz file or an unzipped directory containing .npy files."""
    if os.path.isfile(path) and path.endswith('.npz'):
        d = np.load(path)
        return d['X'], d['Y'], d['M']
    elif os.path.isdir(path):
        x_path = os.path.join(path, 'X.npy')
        y_path = os.path.join(path, 'Y.npy')
        m_path = os.path.join(path, 'M.npy')
        if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(m_path)):
            raise FileNotFoundError(f"Missing X.npy, Y.npy, or M.npy inside directory {path}")
        return np.load(x_path), np.load(y_path), np.load(m_path)
    else:
        raise ValueError(f"Unrecognized shard path format: {path}")

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    shard_targets = [
        os.path.join(root, 'rank_shard_0.npz'),
        os.path.join(root, 'rank_shard_1'),
        os.path.join(root, 'rank_shard_2')
    ]

    print("Locating Stage-C shards:")
    Xs, Ys, Ms = [], [], []
    for idx, target in enumerate(shard_targets):
        if not os.path.exists(target):
            raise FileNotFoundError(f"Shard {idx} not found at {target}")
        x, y, m = load_shard(target)
        print(f"  [Shard {idx}] {os.path.basename(target)}: X={x.shape}, Y={y.shape} (positives: {int(y.sum()):,}), M={m.shape} (M=0: {int((m==0).sum()):,})")
        Xs.append(x)
        Ys.append(y)
        Ms.append(m)

    # Vertical concatenation
    X = np.vstack(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.int8)
    M = np.concatenate(Ms).astype(np.int8)

    # Exact assertions from casmi_rank_matrix.ipynb
    assert X.shape[1] == 31, f"Expected 31 features, got {X.shape[1]}"
    assert np.isfinite(X).all(), "Error: Non-finite values found in feature matrix X"
    assert set(np.unique(Y)) <= {0, 1} and Y.sum() > 0, "Error: Invalid target labels in Y"
    frac1 = float((M == 0).mean())
    print(f"\n[Combined Matrix]")
    print(f"  Shape: X={X.shape}, Y={Y.shape}, M={M.shape}")
    print(f"  Positives: {int(Y.sum()):,} ({Y.mean() * 100:.2f}%)")
    print(f"  Class-1 (M=0) fraction: {frac1:.2f}")
    assert 0.35 < frac1 < 0.65, f"M split is {frac1:.2f}; CFG.W1_PRIORS hedging assumes a balanced split"

    out_path = os.path.join(root, 'rank_train.npz')
    np.savez_compressed(out_path, X=X, Y=Y, M=M)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully written: {out_path} ({size_mb:.2f} MB)")

    # Sanity-check ranker fit (same test as phase1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    W = np.where(M == 0, 0.30, 0.70)
    m0 = HistGradientBoostingClassifier(random_state=0, max_iter=100, min_samples_leaf=20, l2_regularization=1.0).fit(X, Y, sample_weight=W)
    p = m0.predict_proba(X)[:, 1]
    print(f"  Ranker separation: mean p(positive)={p[Y==1].mean():.3f} vs mean p(negative)={p[Y==0].mean():.3f}")

if __name__ == '__main__':
    main()
