"""R3: spectrum -> fingerprint channel (FPNet). Candidate score f.z = candidate bits . predicted logits.

Two checkpoint families with different architectures and bit sets; never mix them:
- "own" (family P): build/fpnet_train.py / prvsiyan's phase1 FPNet (6 layers, adduct + instrument embeddings,
  128 peaks). Raw bits Morgan r2 4096 + r3 4096 + RDKitFP(2048, maxPath 6) + MACCS 167 = 10,407; fp_bits.npy
  (next to coco_fp.npy) selects 6,930. Models whose basename contains "merged" score the merged pseudo-spectrum,
  the rest score each spectrum; z = mean(single-model per-spectrum mean, merged-model merged logits).
  Copied from the zenith Engine 2 `pv_fp.molecule_logits`.
- "prv" (family P, R4): prvsiyan's casmi26-fp-models-v2 (fp_single_s2, fp_merged_m1), the zenith Engine 2 models
  themselves; same model, featuriser and fp_bits.npy as "own", found by dataset path (no fp_bits_sha1.txt).
- "ahmed" (family A): Ahmed Berat Ozer's casmi26-v4b engine FPNet (casmi/fpnet.py + casmi/chem.py, vendored here
  with attribution; 8 layers, ADDUCT_LIST embedding, ce_known / merged / mode inputs, 160 peaks). Raw bits
  ecfp4 4096, ecfp6 4096, fcfp4 2048, rdk6 2048, ap 2048, tt 2048, maccs 167 = 16,507; casmi26-v2-pool's
  fp_bits.npy selects 10,226. z = 0.5 (mean per-spectrum logits + mean per-polarity merged logits), at most 16
  spectra, fp16 autocast on T4. Copied from the zenith PubChem runner `molecule_logits`.

torch / RDKit are imported lazily so the rest of the package (and the tests) never need them.
"""
import hashlib
import math
import os

import numpy as np

# ---------------------------------------------------------------------------------------------- shared helpers
_TORCH = {}


def _torch():
    if not _TORCH:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _TORCH.update(torch=torch, nn=nn, F=F)
    return _TORCH["torch"], _TORCH["nn"], _TORCH["F"]


def bits_sha1(bits):
    return hashlib.sha1(np.ascontiguousarray(bits).tobytes()).hexdigest()


def _ce_scalar(v):
    """(mean |CE|, mean CE, n) of a list-like collision energy; NaN / 0 when absent. Family P uses |CE| (zenith
    parse_ce), family A the signed mean (casmi/build.py ce_mean)."""
    try:
        a = np.atleast_1d(np.asarray(v, float)) if v is not None else np.zeros(0)
        a = a[np.isfinite(a)]
        return (float(np.abs(a).mean()), float(a.mean()), int(len(a))) if len(a) else (np.nan, np.nan, 0)
    except Exception:
        return np.nan, np.nan, 0


def molecule_spectra(sub):
    """One molecule's rows -> spectrum dicts (raw peaks; both families clean them their own way)."""
    out = []
    has_instr = "instrument_type" in sub.columns
    for r in sub.itertuples():
        ce, ce_signed, ce_n = _ce_scalar(r.collision_energy_ev)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        prec=float(r.precursor_mz), adduct=str(r.adduct),
                        mode=1 if r.ionization_mode == "positive" else -1,
                        instr=str(r.instrument_type) if has_instr else None, ce=ce, ce_signed=ce_signed,
                        ce_n=ce_n))
    return out


def _merge_peaks(pairs, mz_tol=0.005, chain=False):
    """Collapse several (mz, it) lists into one (each normalised to base peak 1; max wins on near-duplicates).
    chain=False: pairwise neighbour rule (family P `merge_peaks`); chain=True: family A `merge_spectra`."""
    pairs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in pairs if len(a)]
    pairs = [(a, b) for a, b in pairs if b.max() > 0] if chain else pairs
    if not pairs:
        return np.zeros(0), np.zeros(0)
    mz = np.concatenate([a for a, _ in pairs])
    it = np.concatenate([b / max(float(b.max()), 1e-9) for _, b in pairs])
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    last = 0
    for j in range(1, len(mz)):
        ref = last if chain else j - 1
        if mz[j] - mz[ref] < mz_tol:
            if it[j] >= it[ref]:
                keep[ref] = False
                last = j
            else:
                keep[j] = False
        else:
            last = j
    return mz[keep], it[keep]


def _sin_emb_cls(hi):
    torch, nn, _ = _torch()

    class SinEmb(nn.Module):
        def __init__(self, dim, lo=-2.0, hi=hi, power=1.0):
            super().__init__()
            n = dim // 2
            wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
            self.register_buffer("inv", (2 * math.pi) / wav)

        def forward(self, x):
            a = x.unsqueeze(-1) * self.inv
            return torch.cat([torch.sin(a), torch.cos(a)], -1)

    return SinEmb


def _block_cls():
    torch, nn, F = _torch()

    class Block(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__()
            self.h = h
            self.n1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, pad):
            B, N, D = x.shape
            y = self.n1(x)
            q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :])
            x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
            return x + self.drop(self.ff(self.n2(x)))

    return Block


def _pool_head(x, pad):
    cls = x[:, 0]
    msk = (~pad[:, 1:]).float().unsqueeze(-1)
    mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
    return cls, mean


# ---------------------------------------------------------------------------------------------- family P ("own")
P_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
             "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
             "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
             "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
P_AD_IX = {a: i for i, a in enumerate(P_ADDUCTS)}
P_INSTR = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
P_RAW_BITS = 4096 + 4096 + 2048 + 167
P_KINDS = ("own", "prv")    # family P banks: our casmi-fpnet export, prvsiyan casmi26-fp-models-v2


def p_instr_family(s):
    if s is None:
        return 4
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if "orbitrap" in t or "qft" in t or "ftms" in t or "hybrid ft" in t or "itft" in t or "exactive" in t:
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qq" in t:
        return 3
    return 4


def p_prep_peaks(mz, inten, prec_mz, max_peaks=128, floor=1e-3, win=50.0, per_win=8):
    """Verbatim family-P peak prep (<= prec + 1.5, floor, at most 8 per 50 Da, top 128, sqrt intensity)."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt, sel = {}, []
        for i in order:
            b = bucket[i]
            c = cnt.get(b, 0)
            if c < per_win:
                cnt[b] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            if len(rest):
                sel = np.concatenate([sel, rest[:max_peaks - len(sel)]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


def p_model(nbits, d=512, layers=6, heads=8, drop=0.1):
    torch, nn, _ = _torch()
    SinEmb, Block = _sin_emb_cls(3.2), _block_cls()

    class FPNetP(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.mz_emb, self.nl_emb = SinEmb(d), SinEmb(d)
            self.pk = nn.Linear(2 * d + 1, d)
            self.prec_emb = SinEmb(d)
            self.ad = nn.Embedding(len(P_ADDUCTS), d)
            self.ins = nn.Embedding(len(P_INSTR), d)
            self.gl = nn.Linear(d + 3, d)
            self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

        def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
            B = mz.shape[0]
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
            g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                                   torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
            x = torch.cat([g.unsqueeze(1), p], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
            for b in self.blocks:
                x = b(x, pad)
            x = self.norm(x)
            return self.head(torch.cat(_pool_head(x, pad), -1))

    return FPNetP()


# ---------------------------------------------------------------------------------------------- family A ("ahmed")
A_TEST_ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                  "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
# key order of casmi/chem.py ADDUCTS (insertion order defines the embedding rows)
A_ADDUCT_KEYS = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M+Li]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M-H2O]+",
                 "[M]+", "[M+2H]2+", "[M+3H]3+", "[M+H+Na]2+", "[M+HNa]2+", "[M+2Na]2+", "[M+2Na-H]+", "[M+Ca]2+",
                 "[M+CH3OH+H]+", "[M+CH3CN+H]+", "[M+CH3CN+Na]+", "[M+C2H4N]+", "[M+CH5O]+", "[M-H2N]+",
                 "[M-NH3+H]+", "[M-H5O3]+", "[M-CH3]+", "[M+H-CH3]+", "[M-H]-", "[M-2H]-", "[M-2H]2-",
                 "[M-H2]2-", "[M]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+HCOO]-", "[M+C2H4O2-H]-", "[M+CH3COO]-",
                 "[M+Cl]-", "[M+Br]-", "[M+Na-2H]-", "[M+K-2H]-", "[M+Na-H2]-", "[M+K-H2]-", "[M-CH3]-",
                 "[M-C3H6NO2]-", "[M+C2F3O2]-", "[M+CF3COO]-", "[2M+H]+", "[2M+Na]+", "[2M+K]+", "[2M+NH4]+",
                 "[2M+Ca]2+", "[2M-H]-", "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[2M+Cl]-", "[3M+H]+",
                 "[3M+Na]+", "[3M-H]-"]
A_ADDUCTS = A_TEST_ADDUCTS + [a for a in A_ADDUCT_KEYS if a not in A_TEST_ADDUCTS] + ["<unk>"]
A_AD_IX = {a: i for i, a in enumerate(A_ADDUCTS)}
A_RAW_BITS = 4096 + 4096 + 2048 + 2048 + 2048 + 2048 + 167
A_MAX_SPECTRA = 16


def a_prep_peaks(mz, it, prec_mz, max_peaks=160, floor=1e-3):
    """Verbatim family-A peak prep (<= prec + 2, floor, top 160 by intensity, sqrt intensity)."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 2.0) & (mz > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0 or it.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    it = it / it.max()
    keep = it >= floor
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        sel = np.argsort(-it)[:max_peaks]
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    return mz[o].astype(np.float32), np.sqrt(it[o]).astype(np.float32)


def a_model(nbits, d=512, layers=8, heads=8, drop=0.1, n_elem=10):
    torch, nn, _ = _torch()
    SinEmb, Block = _sin_emb_cls(3.3), _block_cls()

    class FPNetA(nn.Module):
        def __init__(self):
            super().__init__()
            self.mz_emb, self.nl_emb = SinEmb(d), SinEmb(d)
            self.int_emb = nn.Linear(2, d)
            self.pk = nn.Linear(3 * d, d)
            self.prec_emb = SinEmb(d)
            self.ad = nn.Embedding(len(A_ADDUCTS), d)
            self.ce_emb = SinEmb(d // 4, lo=0.0, hi=2.2)
            self.gl = nn.Linear(d + d // 4 + 4, d)
            self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))
            self.elem_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, n_elem))

        def forward(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode):
            B = mz.shape[0]
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl),
                                   self.int_emb(torch.stack([it, it * it], -1))], -1))
            g = self.gl(torch.cat([self.prec_emb(prec), self.ce_emb(ce / 10.0 + 1.0), ce_known.unsqueeze(-1),
                                   merged.unsqueeze(-1), mode.unsqueeze(-1),
                                   torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad)
            x = torch.cat([g.unsqueeze(1), p], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
            for b in self.blocks:
                x = b(x, pad)
            x = self.norm(x)
            return self.head(torch.cat(_pool_head(x, pad), -1))

    return FPNetA()


# ---------------------------------------------------------------------------------------------- featurisers
class Featuriser:
    """SMILES -> selected binary bits (uint8, n x nbits) for one family; RDKit generators built once."""

    def __init__(self, kind, bits):
        self.kind, self.bits = kind, np.asarray(bits, np.int64)
        self.nbits = len(self.bits)
        self.raw_bits = P_RAW_BITS if kind in P_KINDS else A_RAW_BITS
        if self.bits.max() >= self.raw_bits:
            raise ValueError(f"fp_bits indexes past the {kind} raw layout ({self.bits.max()} >= {self.raw_bits})")
        from rdkit import Chem, RDLogger
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as g
        RDLogger.DisableLog("rdApp.*")
        self._chem, self._maccs = Chem, MACCSkeys
        if kind in P_KINDS:
            self._gens = [g.GetMorganGenerator(radius=2, fpSize=4096), g.GetMorganGenerator(radius=3, fpSize=4096),
                          g.GetRDKitFPGenerator(fpSize=2048, maxPath=6)]
        else:
            self._gens = [g.GetMorganGenerator(radius=2, fpSize=4096), g.GetMorganGenerator(radius=3, fpSize=4096),
                          g.GetMorganGenerator(radius=2, fpSize=2048,
                                               atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
                          g.GetRDKitFPGenerator(fpSize=2048, maxPath=6), g.GetAtomPairGenerator(fpSize=2048),
                          g.GetTopologicalTorsionGenerator(fpSize=2048)]

    def raw(self, smi):
        try:
            m = self._chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
            if m is None:
                return None
            parts = [gen.GetFingerprintAsNumPy(m).astype(np.uint8) for gen in self._gens]
            parts.append(np.array(list(self._maccs.GenMACCSKeys(m)), dtype=np.uint8))
            return np.concatenate(parts)
        except Exception:
            return None

    def __call__(self, smiles):
        """-> (bits (n, nbits) uint8, ok (n,) bool)."""
        out = np.zeros((len(smiles), self.nbits), np.uint8)
        ok = np.zeros(len(smiles), bool)
        for i, s in enumerate(smiles):
            r = self.raw(s)
            if r is not None:
                out[i], ok[i] = r[self.bits], True
        return out, ok


def featuriser_parity(feat, smiles, packed_ref):
    """Rows where our bits equal a precomputed packed reference (coco_fp.npy / pool_fp.npy rows)."""
    bits, ok = feat(smiles)
    ref = np.unpackbits(np.asarray(packed_ref, np.uint8), axis=1)[:, :feat.nbits]
    return int(sum(bool(o) and np.array_equal(b, r) for b, r, o in zip(bits, ref, ok)))


# ---------------------------------------------------------------------------------------------- banks
class FPBank:
    """Loaded checkpoints of one family + its featuriser. `molecule_z(spectra)` -> logits (nbits,) or None;
    `fz(z, keys, smiles)` -> f.z per candidate (NaN when the SMILES does not parse)."""

    def __init__(self, kind, models, merged, feat, device, sha1=None, paths=()):
        self.kind, self.models, self.merged, self.feat, self.device = kind, models, merged, feat, device
        self.nbits, self.sha1, self.paths = feat.nbits, sha1, list(paths)
        self._cache = {}

    def describe(self):
        return dict(kind=self.kind, nbits=self.nbits, fp_bits_sha1=self.sha1, device=self.device,
                    models=[os.path.basename(p) for p in self.paths],
                    merged=[os.path.basename(p) for p, m in zip(self.paths, self.merged) if m])

    # -- logits
    def _run(self, nets, items, args_fn, autocast=False):
        torch, _, _ = _torch()
        items = [x for x in items if len(x["mz"])]
        if not items or not nets:
            return None
        B, N = len(items), max(len(x["mz"]) for x in items)
        mz = np.zeros((B, N), np.float32)
        it = np.zeros((B, N), np.float32)
        pad = np.ones((B, N), bool)
        for i, x in enumerate(items):
            n = len(x["mz"])
            mz[i, :n], it[i, :n], pad[i, :n] = x["mz"], x["it"], False
        T = lambda a, dt=None: torch.as_tensor(np.asarray(a, dt), device=self.device)  # noqa: E731
        args = (T(mz), T(it), T(pad)) + args_fn(items, T)
        use_ac = autocast and self.device == "cuda"
        dtype = torch.float16 if use_ac and torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_ac):
            return np.stack([n(*args).float().cpu().numpy() for n in nets]).mean(0)   # (B, nbits)

    def molecule_z(self, spectra):
        if not spectra:
            return None
        return self._z_own(spectra) if self.kind in P_KINDS else self._z_ahmed(spectra)

    def _z_own(self, spectra):
        single = [m for m, g in zip(self.models, self.merged) if not g]
        merged = [m for m, g in zip(self.models, self.merged) if g]
        f32 = np.float32

        def args(items, T):
            return (T([x["prec"] for x in items], f32), T([P_AD_IX.get(x["adduct"], P_AD_IX["<unk>"]) for x in items]),
                    T([p_instr_family(x["instr"]) for x in items]), T([x["ce"] for x in items], f32),
                    T([x["mode"] for x in items], f32))

        out = []
        items = []
        for s in spectra:
            mz, it = self._clip(s)
            pm, pi = p_prep_peaks(mz, it, s["prec"])
            items.append(dict(mz=pm, it=pi, prec=s["prec"], adduct=s["adduct"], instr=s["instr"],
                              ce=s["ce"] if np.isfinite(s["ce"]) else 25.0, mode=float(s["mode"])))
        za = self._run(single, items, args)
        if za is not None:
            out.append(za.mean(0))
        if merged:
            mz, it = _merge_peaks([self._clip(s) for s in spectra])
            pm = float(np.median([s["prec"] for s in spectra]))
            a, b = p_prep_peaks(mz, it, pm)
            zb = self._run(merged, [dict(mz=a, it=b, prec=pm, adduct=spectra[0]["adduct"], instr=spectra[0]["instr"],
                                         ce=25.0, mode=float(np.mean([s["mode"] for s in spectra])))], args)
            if zb is not None:
                out.append(zb.mean(0))
        return np.mean(out, axis=0).astype(np.float32) if out else None

    @staticmethod
    def _clip(s):
        """Zenith Engine 2 input: peaks <= prec + 2, base-peak normalised."""
        k = s["mz"] <= s["prec"] + 2.0
        a, b = s["mz"][k], s["it"][k]
        return a, (b / b.max() if len(b) and b.max() > 0 else b)

    def _z_ahmed(self, spectra):
        spectra = spectra[:A_MAX_SPECTRA]
        f32 = np.float32

        def args(items, T):
            return (T([x["prec"] for x in items], f32), T([x["adduct_ix"] for x in items]),
                    T([x["ce"] for x in items], f32), T([x["ce_known"] for x in items], f32),
                    T([x["n_merged"] / 4.0 for x in items], f32), T([x["mode"] for x in items], f32))

        items_s = []
        for s in spectra:
            pm, pi = a_prep_peaks(s["mz"], s["it"], s["prec"])
            ok = np.isfinite(s["ce_signed"])
            items_s.append(dict(mz=pm, it=pi, prec=s["prec"], adduct_ix=A_AD_IX.get(s["adduct"], A_AD_IX["<unk>"]),
                                ce=s["ce_signed"] if ok else 0.0, ce_known=1.0 if ok else 0.0,
                                n_merged=min(max(1, s["ce_n"]), 8), mode=float(s["mode"])))
        items_m = []
        for md in (1, -1):
            grp = [s for s in spectra if s["mode"] == md]
            if not grp:
                continue
            mm, ii = _merge_peaks([(s["mz"], s["it"]) for s in grp], chain=True)
            prec = float(np.median([s["prec"] for s in grp]))
            pm, pi = a_prep_peaks(mm, ii, prec)
            ces = [s["ce_signed"] for s in grp if np.isfinite(s["ce_signed"])]
            ads = [s["adduct"] for s in grp]
            ad = max(ads, key=ads.count)
            items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=A_AD_IX.get(ad, A_AD_IX["<unk>"]),
                                ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                                n_merged=min(sum(max(1, s["ce_n"]) for s in grp), 8), mode=float(md)))
        zs = self._run(self.models, items_s, args, autocast=True)
        zm = self._run(self.models, items_m, args, autocast=True)
        if zs is None:
            return None
        return (0.5 * (zs.mean(0) + (zm.mean(0) if zm is not None else zs.mean(0)))).astype(np.float32)

    # -- candidate scores
    def fz(self, z, keys, smiles):
        out = np.full(len(keys), np.nan, np.float32)
        if z is None or not len(keys):
            return out
        miss = [i for i, k in enumerate(keys) if k not in self._cache]
        if miss:
            if len(self._cache) > 300_000:
                self._cache.clear()
            bits, ok = self.feat([smiles[i] for i in miss])
            for j, i in enumerate(miss):
                self._cache[keys[i]] = np.packbits(bits[j]) if ok[j] else None
        z = np.asarray(z, np.float32)
        for i, k in enumerate(keys):
            p = self._cache.get(k)
            if p is not None:
                out[i] = float(np.unpackbits(p)[:self.nbits] @ z)
        return out

    def release(self):
        """Free the GPU after pass 1 (z is cached in the plans; f.z is numpy)."""
        self.models = []
        try:
            torch, _, _ = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_bank(kind, paths, bits_path, device=None):
    """kind "own": family P checkpoints (fp_*.pt; 'merged' in the basename -> merged model), fp_bits sha1 asserted
    against each checkpoint's fp_bits_sha1 when present. kind "prv": prvsiyan's family-P set (same model and
    featuriser, no sha1). kind "ahmed": family A checkpoints (fpnet_*.pt)."""
    torch, _, _ = _torch()
    if kind not in P_KINDS + ("ahmed",):
        raise ValueError(kind)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bits = np.load(bits_path)
    sha = bits_sha1(bits)
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no {kind} FPNet checkpoints")
    models, merged = [], []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if int(ck["nbits"]) != len(bits):
            raise ValueError(f"{p}: nbits {ck['nbits']} != fp_bits {len(bits)}")
        if kind in P_KINDS:
            if ck.get("fp_bits_sha1") not in (None, sha):
                raise ValueError(f"{p}: fp_bits sha1 {ck['fp_bits_sha1']} != {sha}")
            net = p_model(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
        else:
            if ck.get("arch", "fpnet") != "fpnet":
                raise ValueError(f"{p}: arch {ck.get('arch')} not supported")
            n_ad = ck["model"]["ad.weight"].shape[0]
            if n_ad != len(A_ADDUCTS):
                raise ValueError(f"{p}: adduct embedding {n_ad} != vendored ADDUCT_LIST {len(A_ADDUCTS)}")
            net = a_model(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]), heads=int(ck.get("heads", 8)))
        net.load_state_dict(ck["model"])
        models.append(net.to(device).eval())
        merged.append(kind in P_KINDS and "merged" in os.path.basename(p))
    if kind in P_KINDS and not (any(merged) and not all(merged)):
        raise ValueError(f"{kind} bank needs at least one single and one merged checkpoint")
    return FPBank(kind, models, merged, Featuriser(kind, bits), device, sha1=sha, paths=paths)


# ---------------------------------------------------------------------------------------------- dataset discovery
OWN_CKPT_GLOB = "**/fp_*.pt"                 # our FPNet export: fp_single_s0/s1, fp_merged_m0/m1 + fp_bits_sha1.txt
AHMED_CKPT_GLOB = "**/fpnet_full1.pt"        # casmi26-fpnet-full1
PRV_DATASET = "casmi26-fp-models-v2"         # prvsiyan: fp_single_s2.pt, fp_merged_m1.pt (no fp_bits_sha1.txt)


def own_ckpt_paths(input_dir):
    """Our FPNet export, found by content, not by dataset name: the folder whose fp_*.pt sit next to
    fp_bits_sha1.txt (written by build/fpnet_train.py; prvsiyan's casmi26-fp-models-v2 has no such file)."""
    import glob
    dirs = {}
    for p in glob.glob(os.path.join(input_dir, OWN_CKPT_GLOB), recursive=True):
        dirs.setdefault(os.path.dirname(p), []).append(p)
    ours = sorted((d for d in dirs if os.path.exists(os.path.join(d, "fp_bits_sha1.txt"))), key=len)
    return sorted(dirs[ours[0]]) if ours else []


def find_bank_files(kind, input_dir, bits_dirs=None):
    """-> (checkpoint paths, fp_bits path, parity reference dir). The fp_bits.npy is chosen by content, not by
    dataset name: for "own" the one whose sha1 equals the checkpoints' fp_bits_sha1 (it sits next to coco_fp.npy);
    for "prv" the one next to coco_fp.npy whose length equals the checkpoint's nbits (as zenith's Engine 2);
    for "ahmed" the one whose length equals the checkpoint's nbits (casmi26-v2-pool, next to pool_fp.npy).
    bits_dirs: where to look for fp_bits.npy (default: input_dir, recursively)."""
    import glob
    torch, _, _ = _torch()
    if kind == "own":
        paths = own_ckpt_paths(input_dir)
    elif kind == "prv":
        paths = sorted(p for p in glob.glob(os.path.join(input_dir, OWN_CKPT_GLOB), recursive=True)
                       if PRV_DATASET in p)                       # as zenith Engine 2
    else:
        paths = sorted(glob.glob(os.path.join(input_dir, AHMED_CKPT_GLOB), recursive=True))[:1]
    if not paths:
        raise FileNotFoundError(f"{kind}: no FPNet checkpoints under {input_dir} -- attach "
                                + {"own": "our FPNet export (fp_*.pt + fp_bits_sha1.txt, e.g. casmi-fpnet)",
                                   "prv": f"prvsiyan/{PRV_DATASET} (fp_single_s2.pt, fp_merged_m1.pt)"}.get(
                                    kind, "casmi26-fpnet-full1 (fpnet_full1.pt)"))
    ck = torch.load(paths[0], map_location="cpu", weights_only=False)
    want_sha, want_n = ck.get("fp_bits_sha1"), int(ck["nbits"])
    txt = os.path.join(os.path.dirname(paths[0]), "fp_bits_sha1.txt")
    if kind == "own" and not want_sha and os.path.exists(txt):
        want_sha = open(txt).read().strip()
    cands = [h for d in (bits_dirs or [input_dir])
             for h in glob.glob(os.path.join(str(d), "**/fp_bits.npy"), recursive=True)]
    for b in sorted(cands, key=len):
        bits = np.load(b)
        if len(bits) != want_n or (kind == "own" and want_sha and bits_sha1(bits) != want_sha):
            continue
        if kind == "prv" and not os.path.exists(os.path.join(os.path.dirname(b), "coco_fp.npy")):
            continue
        return paths, b, os.path.dirname(b)
    raise FileNotFoundError(f"{kind}: no fp_bits.npy with nbits {want_n}" + (f", sha1 {want_sha}" if want_sha else "")
                            + " -- attach the COCONUT arrays dataset with fp_bits.npy + coco_fp.npy (own, prv) or "
                            "casmi26-v2-pool (ahmed)")


def parity_reference(kind, ref_dir, n=50, seed=0):
    """(smiles, packed bits) of n random precomputed rows next to fp_bits.npy: coco_meta.pkl + coco_fp.npy (own),
    pool_meta.parquet + pool_fp.npy (ahmed). None when the files are absent."""
    rng = np.random.default_rng(seed)
    if kind in P_KINDS:
        import pickle
        meta, fp = os.path.join(ref_dir, "coco_meta.pkl"), os.path.join(ref_dir, "coco_fp.npy")
        if not (os.path.exists(meta) and os.path.exists(fp)):
            return None
        smiles = np.asarray(pickle.load(open(meta, "rb"))["smiles"], dtype=object)
    else:
        import pandas as pd
        meta, fp = os.path.join(ref_dir, "pool_meta.parquet"), os.path.join(ref_dir, "pool_fp.npy")
        if not (os.path.exists(meta) and os.path.exists(fp)):
            return None
        smiles = pd.read_parquet(meta, columns=["smiles"]).smiles.to_numpy(dtype=object)
    packed = np.load(fp, mmap_mode="r")
    if len(packed) != len(smiles):
        return None
    idx = np.sort(rng.choice(len(smiles), size=min(n, len(smiles)), replace=False))
    return list(smiles[idx]), np.asarray(packed[idx])
