"""DreaMS embedding inference, vendored: torch + numpy only (specs_novel plan §1.3 A2, roadmap P0.1).

The upstream package (pluskal-lab/DreaMS @ dbec3a0, local `DreamS/`) cannot be imported on Kaggle:
its pins have no Python 3.12 wheels, `dreams.api` pulls in wandb / pyopenms / umap / igraph, and its
checkpoints pickle `dreams.*` classes. So the network code needed for the forward pass is copied here
with the SAME module and attribute names, which makes the upstream `state_dict` keys load `strict`.

    DreamS/dreams/models/layers/fourier_features.py   FourierFeatures            (verbatim)
    DreamS/dreams/models/layers/feed_forward.py       FeedForward                (verbatim; interpolate_interval inlined)
    DreamS/dreams/models/dreams/layers.py             MultiheadAttention, TFeedForward (upstream
                                                      name FeedForward), ScaleNorm, TransformerEncoder
    DreamS/dreams/models/dreams/dreams.py             DreaMSBackbone: DreaMS.__init__ / forward /
                                                      __normalize_spec, inference branches only
    DreamS/dreams/models/heads/heads.py               ContrastiveHead: head(backbone(spec)[:, 0])

The weights come from `dreams_embed.pt`, written once on Colab by the P0.1a conversion (the real
package loads `embedding_model.ckpt` + `ssl_model.ckpt`; `args_to_dict` + `state_dict` are saved).
File layout: {"args_json": str, "state_dict": {name: tensor}, "meta_json": str}; plain types only, so
`torch.load(weights_only=True)` works.

Preprocessing is split in two, exactly as the cached corpus was built:
    dreams_peaks     casmi_dreams_colab.ipynb cell 19 `_dreams_peaks` (= cell 7 `peak_list`): +2 Da cut,
                     0.2 % floor, top-100, m/z ascending, intensity / max   -> (2, n) float32
    to_model_input   upstream SpectrumPreprocessor(DataFormatA, n_highest_peaks=100) as used by
                     `dreams_embeddings`: trim, zero-pad to 100, relative intensity, prepend
                     [precursor_mz, 1.1], float32                           -> (101, 2) float32
m/z above 1000 is NOT clipped (upstream never sets normalize_mzs); it is only outside pretraining.
"""
import hashlib
import json
from math import ceil
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

N_PEAKS = 100             # checkpoint's n_highest_peaks (dreams/api.py `from_name`)
PREC_INTENS = 1.1         # SpectrumPreprocessor default
INT_FLOOR = 0.002         # casmi_dreams_colab.ipynb CFG
DROP_ABOVE_PRECURSOR = 2.0
MAX_MZ = 1000.0           # dreams/utils/dformats.py DataFormatA.max_mz
MAX_TBXIC_STDEV = 1e-4    # DataFormatA.max_tbxic_stdev
TRAINING_ONLY_PREFIXES = ("backbone.ff_out", "backbone.ff_out_intens", "backbone.mz_masking_loss",
                          "backbone.ro_out")

# Backbone args the inference path reads (dreams.py __init__, layers.py). Anything else is ignored.
ARG_KEYS = ("n_layers", "n_heads", "d_fourier", "d_peak", "d_mz_token", "charge_feature",
            "fourier_strategy", "fourier_num_freqs", "fourier_min_freq", "fourier_trainable",
            "dropout", "ff_fourier_depth", "ff_fourier_d", "no_ffs_bias", "ff_peak_depth",
            "vanilla_transformer", "graphormer_mz_diffs", "graphormer_parametrized", "att_dropout",
            "no_transformer_bias", "attn_mech", "ff_dropout", "residual_dropout", "pre_norm", "scnorm",
            "dformat_max_mz", "dformat_max_tbxic_stdev")


def sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------------------------- #

def dreams_peaks(mz, it, prec, floor=INT_FLOOR, n=N_PEAKS, above=DROP_ABOVE_PRECURSOR):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(mz) == 0: return None
    k = mz <= prec + above; mz, it = mz[k], it[k]
    if len(mz) == 0: return None
    mx = it.max()
    if not np.isfinite(mx) or mx <= 0: return None
    k = it >= floor*mx; mz, it = mz[k], it[k]
    if len(mz) < 2: return None
    # Deterministic where cell 7 was not: its np.argsort (unstable) let the numpy build / CPU decide which
    # tied peaks survive the top-n cut (G0.1-R). Ties at the cut go to the higher m/z; order is by m/z,
    # then intensity, so the output does not depend on the input order either.
    if len(mz) > n:
        keep = np.lexsort((mz, it))[-n:]; mz, it = mz[keep], it[keep]
    o = np.lexsort((it, mz)); mz, it = mz[o], it[o]
    return np.stack([mz, it/it.max()]).astype(np.float32)   # (2, n)


def to_model_input(peaks, prec_mz, n=N_PEAKS, prec_intens=PREC_INTENS):
    """(2, k) peak list -> (n + 1, 2) float32: upstream SpectrumPreprocessor.__call__ with its defaults."""
    pl = np.array(peaks, copy=True)                                  # (2, k)
    keep = np.sort(np.lexsort((pl[0], pl[1]))[-n:])                  # su.trim_peak_list, ties -> higher m/z
    pl = pl[:, keep]
    if pl.shape[1] < n:                                              # su.pad_peak_list
        pl = np.pad(pl, ((0, 0), (0, n - pl.shape[1])), mode="constant", constant_values=0)
    pl = np.array([pl[0], pl[1] / pl[1].max()])                      # su.to_rel_intensity
    spec = np.vstack([np.array([prec_mz, prec_intens]), pl.T])       # su.prepend_precursor_peak
    return spec.astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------------- #
# Network (upstream code; see the module docstring for the file of origin)
# --------------------------------------------------------------------------------------------- #

def interpolate_interval(a, b, n, only_inter=False, rounded=False):
    x_min, x_max = min(a, b), max(a, b)
    res = [x_min + i * (x_max - x_min) / (n + 1) for i in range(1 if only_inter else 0, n + 1 if only_inter else n + 2)]
    if x_max == a:
        res.reverse()
    if rounded:
        res = [round(x) for x in res]
    return res


class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=True, funcs='both', sigma=10, num_freqs=512):

        assert strategy in {'random', 'voronov_et_al', 'lin_float_int'}
        assert funcs in {'both', 'sin', 'cos'}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == 'random':
            self.b = torch.randn(num_freqs) * sigma
        if self.strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)],
            )
        elif self.strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, ceil(x_max), 1)],
            )
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('Fourier frequencies', self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]


class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()

        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == 'interpolated':
            assert depth is not None
            hidden_dim = interpolate_interval(a=in_dim, b=out_dim, n=depth - 1, only_inter=True, rounded=True)
        elif isinstance(hidden_dim, Sequence):  # e.g. is List or Tuple
            depth = len(hidden_dim)
        else:
            raise ValueError

        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = args.att_dropout
        self.use_transformer_bias = not args.no_transformer_bias
        self.attn_mech = args.attn_mech
        self.d_graphormer_params = args.d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError('Required: d_model % n_heads == 0.')

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        bs, n, d = q.size()

        def _split_heads(tensor):
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech == 'additive_v' or self.attn_mech == 'additive_fixed':
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = (att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)


class TFeedForward(nn.Module):
    """Upstream `dreams.models.dreams.layers.FeedForward` (renamed: clashes with the peak FeedForward)."""
    def __init__(self, args):
        super(TFeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        self.use_transformer_bias = not args.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.residual_dropout = args.residual_dropout
        self.n_layers = args.n_layers
        self.pre_norm = args.pre_norm

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([TFeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def _layer_forward(self, i, x, src_mask, graphormer_dists):
        pre_norm = self.pre_norm
        post_norm = not pre_norm
        att = self.atts[i]
        ff = self.ffs[i]
        att_scale = self.scales[2 * i]
        ff_scale = self.scales[2 * i + 1]

        residual = x
        x = att_scale(x) if pre_norm else x
        x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = att_scale(x) if post_norm else x

        residual = x
        x = ff_scale(x) if pre_norm else x
        x = ff(x)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = ff_scale(x) if post_norm else x
        return x

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            x = self._layer_forward(i, x, src_mask, graphormer_dists)
        x = self.scales[-1](x) if self.pre_norm else x
        return x


class DreaMSBackbone(nn.Module):
    """Upstream `DreaMS` (LightningModule) reduced to the modules the embedding forward pass uses."""

    def __init__(self, args):
        super().__init__()
        if args.charge_feature:
            raise NotImplementedError("charge_feature=True: the embedding checkpoint is used without charge")
        if args.d_mz_token or not args.d_fourier:
            raise NotImplementedError("only the Fourier-feature m/z encoding is vendored")
        if args.vanilla_transformer:
            raise NotImplementedError("only the DreaMS TransformerEncoder is vendored")
        self.n_layers = args.n_layers
        self.d_fourier = args.d_fourier
        self.d_peak = args.d_peak
        self.d_model = sum(d for d in [args.d_fourier, args.d_peak, args.d_mz_token] if d)
        args.d_model = self.d_model
        self.max_mz = args.dformat_max_mz
        self.graphormer_mz_diffs = args.graphormer_mz_diffs
        if args.graphormer_mz_diffs and args.graphormer_parametrized:
            args.d_graphormer_params = args.d_fourier if args.d_fourier else 1
        else:
            args.d_graphormer_params = 0

        token_dim = 2
        self.fourier_enc = FourierFeatures(strategy=args.fourier_strategy, num_freqs=args.fourier_num_freqs,
                                           x_min=args.dformat_max_tbxic_stdev if not args.fourier_min_freq else args.fourier_min_freq,
                                           x_max=args.dformat_max_mz,
                                           trainable=args.fourier_trainable)
        self.ff_fourier = FeedForward(in_dim=self.fourier_enc.num_features(), out_dim=args.d_fourier, dropout=args.dropout,
                                      depth=args.ff_fourier_depth, hidden_dim=args.ff_fourier_d, bias=not args.no_ffs_bias)
        self.ff_peak = FeedForward(in_dim=token_dim, hidden_dim=args.d_peak, out_dim=args.d_peak, depth=args.ff_peak_depth,
                                   dropout=args.dropout, bias=not args.no_ffs_bias)
        self.transformer_encoder = TransformerEncoder(args)

    def forward(self, spec):
        padding_mask = spec[:, :, 0] == 0
        peak_embs = self.ff_peak(spec / spec.new_tensor([self.max_mz, 1.]))
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        spec = torch.cat([peak_embs, fourier_features], dim=-1)
        graphormer_dists = None
        if self.graphormer_mz_diffs:
            graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
        return self.transformer_encoder(spec, padding_mask, graphormer_dists)


class DreamsEmbedder(nn.Module):
    """Upstream ContrastiveHead at inference: Linear(d_model, d_model) on the precursor token."""

    def __init__(self, args):
        super().__init__()
        self.backbone = DreaMSBackbone(args)
        self.head = nn.Linear(self.backbone.d_model, self.backbone.d_model, bias=True)

    def forward(self, spec):
        return self.head(self.backbone(spec)[:, 0, ...])


# --------------------------------------------------------------------------------------------- #
# Conversion (Colab, with the real package loaded) and loading (anywhere)
# --------------------------------------------------------------------------------------------- #

def args_to_dict(args):
    """Upstream backbone `args` (argparse Namespace; `dformat` is a DataFormatA object) -> plain dict."""
    src = vars(args) if not isinstance(args, dict) else args
    out = {k: src[k] for k in ARG_KEYS if k in src}
    dformat = src.get("dformat")
    out["dformat_max_mz"] = float(getattr(dformat, "max_mz", MAX_MZ))
    out["dformat_max_tbxic_stdev"] = float(getattr(dformat, "max_tbxic_stdev", MAX_TBXIC_STDEV))
    missing = [k for k in ARG_KEYS if k not in out]
    assert not missing, f"backbone args lack {missing}"
    return out


def inference_state_dict(state_dict):
    """Drop the training-only backbone heads that PreTrainedModel deletes after loading (api.py)."""
    return {k: v for k, v in state_dict.items() if not k.startswith(TRAINING_ONLY_PREFIXES)}


def save_dreams(path, args, state_dict, meta):
    torch.save({"args_json": json.dumps(args_to_dict(args), sort_keys=True),
                "state_dict": {k: v.detach().cpu().float().contiguous() for k, v in
                               inference_state_dict(state_dict).items()},
                "meta_json": json.dumps(meta, sort_keys=True)}, path)


def build_embedder(args_dict):
    return DreamsEmbedder(SimpleNamespace(**args_dict))


def load_dreams(path, device="cpu", expect_sha1=None):
    """-> (DreamsEmbedder in eval / float32 on device, meta dict). strict load: every key must match."""
    if expect_sha1 is not None:
        got = sha1(path)
        assert got == expect_sha1, f"{path}: sha1 {got} != expected {expect_sha1}"
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = build_embedder(json.loads(blob["args_json"]))
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.to(device=device, dtype=torch.float32).eval(), json.loads(blob["meta_json"])


@torch.inference_mode()
def dreams_embed(model, peak_lists, precursors, batch_size=64):
    """peak_lists: (2, k) arrays from `dreams_peaks`; precursors: m/z. -> (n, d_model) float32 numpy."""
    dev = next(model.parameters()).device
    out = []
    for lo in range(0, len(peak_lists), batch_size):
        x = np.stack([to_model_input(p, m) for p, m in
                      zip(peak_lists[lo:lo + batch_size], precursors[lo:lo + batch_size])])
        out.append(model(torch.from_numpy(x).to(dev)).float().cpu().numpy())
    d = model.head.out_features
    return np.concatenate(out) if out else np.zeros((0, d), np.float32)
