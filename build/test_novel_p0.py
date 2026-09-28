"""Offline checks for specs_novel P0 (no Colab, no GPU, no DreaMS package, no checkpoints).

1. dreams_peaks == casmi_dreams_colab.ipynb cell 7 `peak_list`, which built the cached corpus
2. to_model_input == upstream SpectrumPreprocessor steps (functions taken from DreamS/dreams/utils/spectra.py)
3. the vendored network == upstream DreamS layers, numerically, with shared weights; state_dict keys
   use upstream's names; save_dreams -> load_dreams round-trip loads strict with weights_only=True
4. TokenTable reads atom / branch / ring / index symbols as selfies does
5. SelfiesVocab ids == a SelfiesTokenizer-style `tokenizers` build; JSON round-trip
6. prep_structure strips stereo + isotopes and classifies multi-component / element exclusions
7. Tracker == reference (selfies.decoder + RDKit) on every prefix: hand-picked molecules, synthetic
   edge cases (skipped atoms, ignored branches, rings to self, bond-order capping), random SELFIES
8. walk_truth: 0 count / grammar / eos violations on every hand-picked molecule; the DBE prune is
   sound on the neutral C/H/N/O/halogen ones and is expected to break on a quaternary ammonium (A8)
"""
import ast
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

UPSTREAM = ROOT / "DreamS" / "dreams"

SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1",            # nicotine (stereo)
    "O=[N+]([O-])c1ccccc1",                                                     # nitro
    "[O-][n+]1ccccc1",                                                          # pyridine N-oxide
    "CS(=O)(=O)Nc1ccccc1", "NS(=O)(=O)c1ccc(N)cc1",                             # sulfonamides
    "OP(=O)(O)OCC(O)CO", "COP(=S)(OC)Oc1ccc([N+](=O)[O-])cc1",                  # phosphate, thiophosphate
    "CC(=O)[O-]", "C[N+](C)(C)C", "FC(F)(F)c1ccc(Cl)cc1Br", "Ic1ccccc1",
    "CCCCCCCCCCCCCCCC(=O)O",                                                    # palmitic acid (acyclic)
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COP(=O)(O)OCC[N+](C)(C)C)OC(=O)CCCCCCC",     # a lipid, long branches
    "C1CCCCCCCCCCCCCCCCCCCC1",                                                   # 21-ring: [Ring2]
    "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",                                           # estradiol skeleton
    "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",  # sucrose
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",                                     # diazepam
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",                              # penicillin G
    "C#Cc1ccccc1", "N#CC(C#N)=C(C#N)C#N", "C=C=C", "O=C=O", "[C-]#[O+]",
    "O=C1NC(=O)C(=O)N1", "c1ccc2c(c1)ccc1ccccc12", "C1=CC2=CC=CC=CC2=C1",      # azulene
    "CC(C)(C)c1cc(O)ccc1O", "O=S(=O)(O)O", "OS(=O)(=O)c1ccccc1",
    "N=C(N)NCCCC(N)C(=O)O", "CC(N)Cc1c[nH]c2ccccc12", "[13CH3]C(=O)O",         # isotope
    "C1CC2CCC1C2", "C12C3C4C1C5C2C3C45",                                       # norbornane, cubane
    "CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(c34)C2=C1",                 # LSD
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",
]


def load_func(tree, name, ns):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), ns)
    return ns[name]


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_spectra(rng):
    out = []
    for n in (0, 1, 2, 5, 40, 100, 101, 350):
        mz = np.sort(rng.uniform(50, 900, n))
        it = rng.uniform(0, 1, n) ** 3
        out.append((mz, it, 600.0))
    out.append((np.array([100.0, 700.0, 800.0]), np.array([1.0, 1.0, 1.0]), 500.0))   # 2 above prec + 2
    out.append((np.array([100.0, 200.0, 300.0]), np.array([1.0, 1e-4, 1e-5]), 500.0))  # < 2 after floor
    out.append((np.array([100.0, 200.0]), np.array([np.nan, 1.0]), 500.0))             # NaN max
    out.append((np.array([100.0, 200.0]), np.array([0.0, 0.0]), 500.0))                # zero max
    out.append((np.array([100.0, 1201.0, 1203.5]), np.array([0.5, 1.0, 0.2]), 1200.0))  # > 1000 m/z
    return out


def test_preprocessing():
    import dreams_infer as di
    nb = json.loads((ROOT / "casmi_dreams_colab.ipynb").read_text(encoding="utf-8"))
    cell7 = "".join(nb["cells"][7]["source"])
    assert "def peak_list(" in cell7, "casmi_dreams_colab.ipynb cell 7 moved"
    cfg = SimpleNamespace(DROP_ABOVE_PRECURSOR=2.0, INT_FLOOR=0.002, N_PEAKS=100)
    peak_list = load_func(ast.parse(cell7), "peak_list", {"np": np, "CFG": cfg})
    snippet = "".join(nb["cells"][19]["source"])
    for key in ("k = mz <= prec + above", "k = it >= floor*mx", "if len(mz) < 2: return None",
                "keep = np.sort(np.argsort(it)[-n:])", "return np.stack([mz, it/it.max()]).astype(np.float32)"):
        assert key in snippet, f"cell 19 _dreams_peaks changed: {key}"

    spectra_src = (UPSTREAM / "utils" / "spectra.py").read_text(encoding="utf-8")
    tree = ast.parse(spectra_src)
    ns = {"np": np}
    trim = load_func(tree, "trim_peak_list", ns)
    pad = load_func(tree, "pad_peak_list", ns)
    rel = load_func(tree, "to_rel_intensity", ns)
    prepend = load_func(tree, "prepend_precursor_peak", ns)

    rng = np.random.default_rng(0)
    n_ok = 0
    for mz, it, prec in synthetic_spectra(rng):
        a, b = di.dreams_peaks(mz, it, prec), peak_list(mz, it, prec)
        assert (a is None) == (b is None), (len(mz), a, b)
        if a is None:
            continue
        n_ok += 1
        assert a.dtype == np.float32 and a.shape == b.shape and np.array_equal(a, b)
        assert a.shape[1] <= 100 and np.all(np.diff(a[0]) >= 0) and a[1].max() == 1.0
        # upstream SpectrumPreprocessor.__call__(spec (2, n), prec_mz, high_form=False), step by step
        spec = a.copy().T
        spec = trim(spec.T, 100).T
        spec = pad(spec.T, target_len=100).T
        spec = rel(spec.T).T
        spec = prepend(spec, prec, 1.1, high=True).astype(np.float32, copy=False)
        x = di.to_model_input(a, prec)
        assert x.shape == (101, 2) and x.dtype == np.float32 and np.array_equal(x, spec)
        assert x[0, 0] == np.float32(prec) and x[0, 1] == np.float32(1.1)
    assert n_ok >= 6, n_ok
    raw = (np.linspace(60, 590, 250), rng.uniform(0.01, 1, 250))            # > 100 peaks straight in
    spec = prepend(rel(pad(trim(np.stack(raw), 100), 100)).T, 600.0, 1.1, high=True).astype(np.float32)
    assert np.array_equal(di.to_model_input(np.stack(raw), 600.0), spec)
    print("  preprocessing: OK")


TEST_ARGS = dict(n_layers=2, n_heads=4, d_fourier=16, d_peak=16, d_mz_token=None, charge_feature=False,
                 fourier_strategy="lin_float_int", fourier_num_freqs=512, fourier_min_freq=None,
                 fourier_trainable=False, dropout=0.1, ff_fourier_depth=3, ff_fourier_d=48,
                 no_ffs_bias=False, ff_peak_depth=2, vanilla_transformer=False, graphormer_mz_diffs=True,
                 graphormer_parametrized=False, att_dropout=0.1, no_transformer_bias=False,
                 attn_mech="dot-product", ff_dropout=0.1, residual_dropout=0.1, pre_norm=True,
                 scnorm=False, dformat_max_mz=1000.0, dformat_max_tbxic_stdev=1e-4)


def test_network():
    try:
        import torch
    except ImportError:
        print("  network: SKIPPED (no torch)")
        return
    import dreams_infer as di
    torch.manual_seed(0)
    up_layers = load_file_module("up_layers", UPSTREAM / "models" / "dreams" / "layers.py")
    up_ff = load_file_module("up_ff", UPSTREAM / "models" / "layers" / "fourier_features.py")
    ff_tree = ast.parse((UPSTREAM / "models" / "layers" / "feed_forward.py").read_text(encoding="utf-8"))
    up_ns = {"nn": torch.nn, "Sequence": __import__("typing").Sequence,
             "utils": SimpleNamespace(interpolate_interval=di.interpolate_interval)}
    cls = next(n for n in ff_tree.body if isinstance(n, ast.ClassDef) and n.name == "FeedForward")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "feed_forward", "exec"), up_ns)
    UpFeedForward = up_ns["FeedForward"]

    for graphormer_param in (False, True):
        args = dict(TEST_ARGS, graphormer_parametrized=graphormer_param)
        ours = di.build_embedder(args).eval()

        # an upstream-shaped module: the attribute names DreaMS.__init__ uses, upstream classes
        a = SimpleNamespace(**args)
        a.d_model = a.d_fourier + a.d_peak
        a.d_graphormer_params = a.d_fourier if (a.graphormer_mz_diffs and a.graphormer_parametrized) else 0
        up = torch.nn.Module()
        up.fourier_enc = up_ff.FourierFeatures(strategy=a.fourier_strategy, num_freqs=a.fourier_num_freqs,
                                               x_min=a.dformat_max_tbxic_stdev, x_max=a.dformat_max_mz,
                                               trainable=a.fourier_trainable)
        up.ff_fourier = UpFeedForward(in_dim=up.fourier_enc.num_features(), out_dim=a.d_fourier,
                                      dropout=a.dropout, depth=a.ff_fourier_depth, hidden_dim=a.ff_fourier_d,
                                      bias=not a.no_ffs_bias)
        up.ff_peak = UpFeedForward(in_dim=2, hidden_dim=a.d_peak, out_dim=a.d_peak, depth=a.ff_peak_depth,
                                   dropout=a.dropout, bias=not a.no_ffs_bias)
        up.transformer_encoder = up_layers.TransformerEncoder(a)
        up_sd = {"backbone." + k: v for k, v in up.state_dict().items()}
        up_sd["head.weight"] = torch.randn(a.d_model, a.d_model) * 0.1
        up_sd["head.bias"] = torch.randn(a.d_model) * 0.1
        up_sd["backbone.ff_out.ff.0.weight"] = torch.zeros(3, 3)            # a training head: dropped
        assert set(di.inference_state_dict(up_sd)) == set(ours.state_dict()), \
            set(di.inference_state_dict(up_sd)) ^ set(ours.state_dict())
        ours.load_state_dict(di.inference_state_dict(up_sd), strict=True)
        up.eval()

        rng = np.random.default_rng(1)
        pls = [di.dreams_peaks(np.sort(rng.uniform(50, 1100, n)), rng.uniform(0, 1, n), 1150.0)
               for n in (3, 20, 80, 150)]
        x = torch.from_numpy(np.stack([di.to_model_input(p, 1150.0) for p in pls]))
        with torch.no_grad():
            # upstream DreaMS.forward (dreams.py:138-185) with the upstream modules
            pad_mask = x[:, :, 0] == 0
            peak = up.ff_peak(x / torch.tensor([a.dformat_max_mz, 1.]))
            four = up.ff_fourier(up.fourier_enc(x[..., [0]]))
            h = up.transformer_encoder(torch.cat([peak, four], dim=-1), pad_mask,
                                       four.unsqueeze(2) - four.unsqueeze(1))
            want = h[:, 0] @ up_sd["head.weight"].T + up_sd["head.bias"]
            got = ours(x)
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
        # the precursor-token embedding ignores how the real peaks are ordered (no positions)
        perm = torch.cat([torch.tensor([0]), 1 + torch.randperm(100)])
        with torch.no_grad():
            assert torch.allclose(ours(x[:, perm]), got, atol=1e-4)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dreams_embed.pt"
            full_args = dict(args, dformat=SimpleNamespace(max_mz=1000.0, max_tbxic_stdev=1e-4),
                             lr=1e-4, train_objective="mask_mz")
            full_args.pop("dformat_max_mz"); full_args.pop("dformat_max_tbxic_stdev")
            di.save_dreams(p, full_args, up_sd, {"dreams_commit": "test"})
            model, meta = di.load_dreams(p, expect_sha1=di.sha1(p))
            assert meta == {"dreams_commit": "test"}
            emb = di.dreams_embed(model, pls, [1150.0] * len(pls), batch_size=3)
            assert emb.shape == (4, 32) and emb.dtype == np.float32
            assert np.allclose(emb, got.numpy(), atol=1e-5)
            try:
                di.load_dreams(p, expect_sha1="0" * 40)
                raise AssertionError("sha1 mismatch not caught")
            except AssertionError as e:
                assert "sha1" in str(e)
    print("  network: OK")


def test_token_table():
    import novel_decoder as nd
    import selfies as sf
    extra = ["[NH1+1]", "[=N+1]", "[O-1]", "[#N]", "[=Ring1]", "[Branch3]", "[=Branch2]", "[H]", "[CH4]"]
    corpus = [nd.prep_structure(s)[2] for s in SMILES]              # as P0.3: corpus ∪ robust alphabet
    vocab = nd.SelfiesVocab.from_alphabet(set(sf.get_semantic_robust_alphabet()) | set(extra)
                                          | sf.get_alphabet_from_selfies([s for s in corpus if s]))
    tt = nd.TokenTable(vocab)
    def row(t):
        i = vocab.tok2id[t]
        return int(tt.kind[i]), tt.elements[tt.elem[i]] if tt.elem[i] >= 0 else None, int(tt.bond[i]), \
            int(tt.cap[i]), int(tt.nidx[i])
    assert row("[C]") == (nd.K_ATOM, "C", 1, 4, 0)
    assert row("[=C]") == (nd.K_ATOM, "C", 2, 4, 0)
    assert row("[#N]") == (nd.K_ATOM, "N", 3, 3, 0)
    assert row("[N+1]") == (nd.K_ATOM, "N", 1, 4, 0)
    assert row("[NH1+1]") == (nd.K_ATOM, "N", 1, 3, 0)
    assert row("[O-1]") == (nd.K_ATOM, "O", 1, 1, 0)
    assert row("[CH4]") == (nd.K_ATOM, "C", 1, 0, 0)
    assert row("[Branch1]") == (nd.K_BRANCH, None, 1, 0, 1)
    assert row("[=Branch2]") == (nd.K_BRANCH, None, 2, 0, 2)
    assert row("[Branch3]") == (nd.K_BRANCH, None, 1, 0, 3)
    assert row("[Ring2]") == (nd.K_RING, None, 1, 0, 2)
    assert row("[=Ring1]") == (nd.K_RING, None, 2, 0, 1)
    for t in (vocab.id2tok[vocab.pad_id], "<s>", "</s>", "<unk>"):
        assert tt.kind[vocab.tok2id[t]] == nd.K_SPECIAL
    assert tt.is_index.sum() == 16 and tt.is_index[vocab.tok2id["[Ring1]"]]
    print("  token table: OK")
    return vocab, tt


def test_vocab(vocab):
    import novel_decoder as nd
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.json"
        vocab.to_json(p)
        v2 = nd.SelfiesVocab.from_json(p)
    assert v2.tok2id == vocab.tok2id and v2.id2tok == vocab.id2tok
    s = "[C][=C][Branch1][C][O][N+1][Ring1][Branch1]"
    assert v2.decode(v2.encode(s)) == s
    try:
        from tokenizers import Tokenizer, models
        from tokenizers.implementations import BaseTokenizer
    except ImportError:
        print("  vocab: OK (tokenizers parity SKIPPED)")
        return
    # massspecgym/models/tokenizers.py:79-85 and SpecialTokensBaseTokenizer.__init__
    alphabet = sorted(t for t in vocab.tok2id if not t.startswith("<"))
    wl = {s: i for i, s in enumerate(alphabet)}
    wl["<unk>"] = len(wl)
    tok = BaseTokenizer(Tokenizer(models.WordLevel(vocab=wl, unk_token="<unk>")))
    tok.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
    assert tok.get_vocab() == vocab.tok2id, "SelfiesTokenizer id layout differs"
    print("  vocab: OK")


def test_prep():
    import novel_decoder as nd
    st, can, s, f = nd.prep_structure("CN1CCC[C@H]1c1cccnc1")
    assert st == "ok" and "@" not in can and f == {"C": 10, "N": 2, "H": 14}
    st, can, s, f = nd.prep_structure("[13CH3]C(=O)O")
    assert st == "ok" and "13" not in can and nd.ik14(can) == nd.ik14("CC(=O)O")
    assert nd.prep_structure("CC.O")[0] == "multi"
    assert nd.prep_structure("C[Si](C)(C)C")[0] == "element"
    assert nd.prep_structure("not a smiles")[0] == "parse"
    n = 0
    for smi in SMILES:
        st, can, s, f = nd.prep_structure(smi)
        assert st == "ok", (smi, st)
        import selfies as sf
        assert nd.ik14(sf.decoder(s)) == nd.ik14(can), smi
        n += 1
    assert nd.parse_formula("C10H14N2") == {"C": 10, "H": 14, "N": 2}
    assert nd.parse_formula("CH4ClBr2") == {"C": 1, "H": 4, "Cl": 1, "Br": 2}
    assert nd.dbe({"C": 6, "H": 6}) == 4 and nd.dbe({"C": 6, "H": 5, "N": 1, "O": 2}) == 5
    assert nd.dbe_prune_applies({"C": 1, "H": 4}) and not nd.dbe_prune_applies({"C": 1, "S": 1})
    print(f"  prep_structure: OK ({n} round-trips)")


def edge_sequences():
    return [
        ["[C]", "[F]", "[C]", "[O]"],                          # main chain ends at F: C, O skipped
        ["[F]", "[C]"],                                        # root F leaves state 1: C bonds once
        ["[C]", "[Branch1]", "[C]", "[O]"],                    # branch at state 4 -> reads [C] as index
        ["[O]", "[C]", "[Branch1]", "[C]", "[F]", "[N]"],
        ["[F]", "[Branch1]", "[C]", "[O]"],                    # branch at state 1: ignored, [C] is an atom
        ["[Ring1]", "[C]", "[C]"],                             # ring at state 0: ignored, no index read
        ["[C]", "[Ring1]", "[C]"],                             # ring to itself (lidx == ridx): skipped
        ["[C]", "[C]", "[C]", "[=Ring1]", "[Ring1]"],         # ring order capped by the free valence
        ["[C]", "[=C]", "[=C]", "[=Ring1]", "[Ring1]"],
        ["[N]", "[=N]", "[#N]"],                               # bond order capped by state
        ["[C]", "[C]", "[Ring1]", "[C]", "[#C]"],              # ring onto an existing bond: order raised
        ["[C]", "[Branch1]", "[Ring1]", "[O]", "[C]", "[C]"],  # branch Q from a [Ring1] index symbol
        ["[C]", "[Branch2]", "[C]", "[C]", "[C]", "[C]", "[C]"],   # Branch2 with a short tail
        ["[C]", "[C]", "[Branch1]"],                           # prefix ends before the index
        ["[C]", "[O]", "[O]", "[O]"],
        ["[C]", "[Branch1]", "[C]", "[=O]", "[Branch1]", "[C]", "[F]", "[N]"],
    ]


def test_tracker(vocab, tt):
    import novel_decoder as nd
    import selfies as sf
    n_prefix = 0
    for seq in edge_sequences():
        bad = nd.tracker_parity(tt, seq)
        assert not bad, (seq, bad)
        n_prefix += len(seq)
    t = nd.Tracker(tt)
    for tok in ["[C]", "[F]"]:
        t = t.push(vocab.tok2id[tok])
    assert not t.legal_mask()[tt.is_atom].any(), "atoms offered after the main chain ended"

    rnd = random.Random(0)
    alphabet = sorted(sf.get_semantic_robust_alphabet())
    for _ in range(400):
        seq = [rnd.choice(alphabet) for _ in range(rnd.randint(1, 30))]
        bad = nd.tracker_parity(tt, seq)
        assert not bad, (seq, bad)
        n_prefix += len(seq)

    for smi in SMILES:
        st, can, s, f = nd.prep_structure(smi)
        toks = list(sf.split_selfies(s))
        bad = nd.tracker_parity(tt, toks)
        assert not bad, (smi, bad)
        n_prefix += len(toks)
    print(f"  tracker parity: OK ({n_prefix} prefixes)")


def test_walk(vocab, tt):
    import novel_decoder as nd
    import selfies as sf
    dbe_broken = []
    for smi in SMILES:
        st, can, s, f = nd.prep_structure(smi)
        toks = list(sf.split_selfies(s))
        bad = nd.walk_truth(tt, toks, f, max_len=200, dbe_prune=False)
        assert not bad, (smi, bad)
        bad = nd.walk_truth(tt, toks, f, max_len=200, dbe_prune=True)
        assert all(r == "dbe" for _, r, _ in bad), (smi, bad)
        if bad:
            dbe_broken.append(smi)
        # the mask must refuse an atom whose element is used up, and </s> before the formula is done
        t = nd.Tracker(tt, f)
        for tok in toks[:-1]:
            t = t.push(vocab.tok2id[tok])
        if tt.kind[vocab.tok2id[toks[-1]]] == nd.K_ATOM:
            assert not t.eos_legal(), smi
    assert "C[N+](C)(C)C" in dbe_broken, dbe_broken
    neutral = [s for s in SMILES if "+" not in s and "-" not in s and "S" not in s and "P" not in s]
    assert not set(neutral) & set(dbe_broken), set(neutral) & set(dbe_broken)
    # wrong formula: one C too few -> the last C is masked ('count'); one C too many -> no </s>
    st, can, s, f = nd.prep_structure("CCO")
    toks = list(sf.split_selfies(s))
    rules = {r for _, r, _ in nd.walk_truth(tt, toks, dict(f, C=f["C"] - 1))}
    assert "count" in rules, rules
    rules = {r for _, r, _ in nd.walk_truth(tt, toks, dict(f, C=f["C"] + 1))}
    assert rules == {"eos"}, rules
    assert {r for _, r, _ in nd.walk_truth(tt, toks, f, max_len=2)} == {"length"}
    print(f"  mask walk: OK (DBE prune breaks on {len(dbe_broken)}: {', '.join(dbe_broken)})")


def main():
    test_preprocessing()
    test_network()
    vocab, tt = test_token_table()
    test_vocab(vocab)
    test_prep()
    test_tracker(vocab, tt)
    test_walk(vocab, tt)
    print("test_novel_p0: OK")


if __name__ == "__main__":
    main()
