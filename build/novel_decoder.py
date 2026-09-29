"""specs_novel single source of truth for the SELFIES side and the decoder (plan §4, §5.5).

    SelfiesVocab        token <-> id, the same ids as MassSpecGym's SelfiesTokenizer (JSON, no `tokenizers`)
    TokenTable          per-token grammar facts, precomputed with selfies' own grammar_rules
    Tracker             incremental SELFIES derivation state -> legal-token mask (plan §4.1-4.2)
    reference_state     the slow truth: selfies.decoder(prefix) + RDKit counts (P0.4 parity)
    walk_truth          mask soundness: every true next token legal, </s> legal at the end (P0.4)
    prep_structure      stereo/isotope strip -> canonical SMILES -> SELFIES (A9, P0.3)

The Tracker replays `selfies.decoder._derive_mol_from_symbols` token by token. That function is
recursive (a branch derives a sub-chain of Q+1 symbols); here its call stack is an explicit list of
frames [state, prev_atom, budget]. Every consumed symbol lowers the budget of every open frame, the
way each caller adds its child's n_derived; a frame closes when its budget reaches 0. A frame whose
state becomes None (no free valence left) swallows the rest of its budget, and in the main chain that
means every later symbol -- SELFIES silently skips those atoms (A7), so the mask never offers them.

Masks (plan §4.2):
    pending index symbols      only the 16 INDEX_ALPHABET symbols (the encoder emits nothing else there)
    atom token, element e      frame active, the atom would be placed, and remaining[e] > 0
    [Branch*]                  frame state > 1 (otherwise selfies ignores it)
    [Ring*]                    frame state > 0
    </s>                       no pending index and every heavy-atom count exactly used
`unsat()` is rings + pi bonds of the molecule the prefix decodes to, with the ring bonds formed the way
`_form_rings_bilocally` forms them at the end (orders capped by the free valence left). It feeds the
optional DBE prune, which is only applied where P0.4 shows it sound (A8).

P3 part (below the P0 part, specs_novel/gate_report.md -> G3):
    DreamsSelfiesDecoder  memory = per-spectrum tokens (+ adduct, + CE bin) plus a formula token,
                          no positional encoding on the memory, tied in/out token embeddings
    search_prefix_seeded  arm A: masked expansion to depth L_SEED, then sampled rollouts per prefix
    search_masked_beam    arm B (alpha = 0) and arm C (alpha > 0, cross-beam scaffold penalty)
    select_top25          "maxmin" (diversity, what G3 is judged on) or "logp"
    max_tanimoto          the G3 metric, on MassSpecGym's fingerprint

Needs numpy + selfies (pinned) for the P0 part; RDKit for reference_state / prep_structure / ik14 /
the selection helpers; torch for the P3 part.
"""
import json
import re

import numpy as np
import selfies as sf
from selfies.constants import INDEX_CODE
from selfies.grammar_rules import (get_index_from_selfies, next_atom_state, next_branch_state,
                                   next_ring_state, process_atom_symbol, process_branch_symbol,
                                   process_ring_symbol)

PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
ALLOWED_ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I")
DBE_SAFE_ELEMENTS = frozenset(("C", "H", "N", "O", "F", "Cl", "Br", "I"))   # no S / P (plan §4.2)
K_SPECIAL, K_ATOM, K_BRANCH, K_RING, K_OTHER = 0, 1, 2, 3, 4
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def check_constraints():
    """The token table and the tracker assume selfies' default semantic constraints."""
    assert sf.get_semantic_constraints() == sf.get_preset_constraints("default"), \
        "selfies semantic constraints are not the default preset"


# --------------------------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------------------------- #

def vocab_from_alphabet(alphabet):
    """SelfiesTokenizer's id layout (massspecgym/models/tokenizers.py:79-85 + add_special_tokens):
    sorted alphabet 0..n-1, <unk> n, then <pad> <s> </s> as the next added tokens."""
    alphabet = sorted(set(alphabet) - {PAD, SOS, EOS, UNK})
    tok2id = {s: i for i, s in enumerate(alphabet)}
    tok2id[UNK] = len(tok2id)
    for s in (PAD, SOS, EOS):
        tok2id[s] = len(tok2id)
    return tok2id


class SelfiesVocab:
    def __init__(self, tok2id):
        ids = sorted(tok2id.values())
        assert ids == list(range(len(ids))), "token ids must be 0..n-1"
        for s in (PAD, SOS, EOS, UNK):
            assert s in tok2id, f"special token {s} missing"
        self.tok2id = dict(tok2id)
        self.id2tok = [None] * len(ids)
        for s, i in self.tok2id.items():
            self.id2tok[i] = s
        self.pad_id, self.sos_id, self.eos_id, self.unk_id = (self.tok2id[s] for s in (PAD, SOS, EOS, UNK))

    def __len__(self):
        return len(self.id2tok)

    @classmethod
    def from_alphabet(cls, alphabet):
        return cls(vocab_from_alphabet(alphabet))

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tok2id, f, sort_keys=True, indent=0)

    def encode(self, selfies_str):
        """SELFIES string -> ids, no <s>/</s>."""
        return [self.tok2id.get(t, self.unk_id) for t in sf.split_selfies(selfies_str)]

    def decode(self, ids):
        """ids -> SELFIES string; special tokens dropped."""
        return "".join(self.id2tok[i] for i in ids if i not in (self.pad_id, self.sos_id, self.eos_id,
                                                                self.unk_id))


# --------------------------------------------------------------------------------------------- #
# Token table
# --------------------------------------------------------------------------------------------- #

class TokenTable:
    """Per-token facts over a vocabulary, from selfies' own grammar_rules (so they follow the pin)."""

    def __init__(self, vocab):
        check_constraints()
        n = len(vocab)
        self.vocab = vocab
        self.kind = np.full(n, K_OTHER, np.int8)
        self.elem = np.full(n, -1, np.int16)           # index into self.elements
        self.bond = np.zeros(n, np.int8)               # requested bond order (atom / branch / ring type)
        self.cap = np.zeros(n, np.int8)                # atom bonding capacity (after explicit H, charge)
        self.nidx = np.zeros(n, np.int8)               # index symbols read after a branch / ring
        self.is_index = np.zeros(n, bool)              # one of the 16 INDEX_ALPHABET symbols
        self.index_code = np.zeros(n, np.int16)
        elements = []
        for i, t in enumerate(vocab.id2tok):
            if t in (PAD, SOS, EOS, UNK):
                self.kind[i] = K_SPECIAL
                continue
            if t in INDEX_CODE:
                self.is_index[i] = True
                self.index_code[i] = INDEX_CODE[t]
            if "ch" == t[-4:-2]:                       # the same dispatch order as the decoder
                out = process_branch_symbol(t)
                if out is not None:
                    self.kind[i], self.bond[i], self.nidx[i] = K_BRANCH, out[0], out[1]
            elif "ng" == t[-4:-2]:
                out = process_ring_symbol(t)
                if out is not None:
                    self.kind[i], self.bond[i], self.nidx[i] = K_RING, out[0], out[1]
            elif "eps" in t:
                pass
            else:
                out = process_atom_symbol(t)
                if out is not None:
                    (bond, _), atom = out
                    if atom.element not in elements:
                        elements.append(atom.element)
                    self.kind[i] = K_ATOM
                    self.elem[i] = elements.index(atom.element)
                    self.bond[i] = bond
                    self.cap[i] = atom.bonding_capacity
        self.elements = tuple(elements)
        self.elem_ix = {e: j for j, e in enumerate(self.elements)}
        self.h_ix = self.elem_ix.get("H", -1)
        self.is_atom = self.kind == K_ATOM
        self.is_branch = self.kind == K_BRANCH
        self.is_ring = self.kind == K_RING
        self.atom_cap_pos = self.is_atom & (self.cap > 0)
        self.elem_safe = np.maximum(self.elem, 0)

    def counts_vector(self, counts):
        """{element: count} -> vector over self.elements, plus the count of elements the vocab lacks."""
        v = np.zeros(len(self.elements), np.int32)
        missing = 0
        for e, c in counts.items():
            if e in self.elem_ix:
                v[self.elem_ix[e]] = c
            elif e != "H":
                missing += c
        return v, missing


# --------------------------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------------------------- #

INF = float("inf")


class Tracker:
    """Persistent SELFIES derivation state. `push` returns a new Tracker; the old one is unchanged.

    formula: {element: count} incl. H, or None (then only counting, no count masks)."""

    __slots__ = ("tt", "frames", "pending", "caps", "bc", "bonds", "rings", "counts", "target",
                 "missing", "n_tokens", "broken")

    def __init__(self, table, formula=None):
        self.tt = table
        self.frames = [[0, None, INF]]          # [state (int | None), prev atom index, budget]
        self.pending = None                     # [kind, n_left, symbols, payload]
        self.caps = []                          # bonding capacity per placed atom
        self.bc = []                            # bond order sum per atom (chain bonds only)
        self.bonds = {}                         # (a, b) a < b -> order (chain bonds only)
        self.rings = []                         # (left atom, right atom, order), formed at the end
        self.counts = np.zeros(len(table.elements), np.int32)
        if formula is None:
            self.target, self.missing = None, 0
        else:
            self.target, self.missing = table.counts_vector(formula)
        self.n_tokens = 0
        self.broken = False

    def _copy(self):
        t = Tracker.__new__(Tracker)
        t.tt = self.tt
        t.frames = [f[:] for f in self.frames]
        t.pending = None if self.pending is None else [self.pending[0], self.pending[1],
                                                       self.pending[2][:], self.pending[3]]
        t.caps, t.bc, t.bonds, t.rings = self.caps[:], self.bc[:], dict(self.bonds), self.rings[:]
        t.counts = self.counts.copy()
        t.target, t.missing, t.n_tokens, t.broken = self.target, self.missing, self.n_tokens, self.broken
        return t

    # -- derivation -------------------------------------------------------------------------- #

    def _consume(self):
        for f in self.frames:
            f[2] -= 1

    def _close_exhausted(self):
        while len(self.frames) > 1 and self.frames[-1][2] <= 0:
            self.frames.pop()

    def _add_atom(self, cap):
        self.caps.append(int(cap))
        self.bc.append(0)
        return len(self.caps) - 1

    def push(self, tok_id):
        t = self._copy()
        t._apply(int(tok_id))
        return t

    def _apply(self, i):
        tt = self.tt
        self.n_tokens += 1
        self._consume()
        if self.pending is not None:                      # an index symbol of a branch / ring
            p = self.pending
            p[2].append(tt.vocab.id2tok[i])
            p[1] -= 1
            if p[1] == 0:
                self.pending = None
                q = get_index_from_selfies(*p[2])
                f = self.frames[-1]
                if p[0] == K_BRANCH:
                    self.frames.append([p[3], f[1], q + 1])
                else:
                    lidx = max(0, f[1] - (q + 1))
                    self.rings.append((lidx, f[1], p[3]))
            self._close_exhausted()
            return
        f = self.frames[-1]
        state = f[0]
        kind = tt.kind[i]
        if state is None or kind in (K_SPECIAL, K_OTHER):
            if kind == K_OTHER and state is not None and "eps" in tt.vocab.id2tok[i]:
                f[0] = 0 if state == 0 else None
            elif kind == K_SPECIAL:
                self.broken = True                        # specials never occur inside a SELFIES string
        elif kind == K_BRANCH:
            if state > 1:
                binit, f[0] = next_branch_state(int(tt.bond[i]), state)
                self.pending = [K_BRANCH, int(tt.nidx[i]), [], binit]
        elif kind == K_RING:
            if state != 0:
                order, f[0] = next_ring_state(int(tt.bond[i]), state)
                self.pending = [K_RING, int(tt.nidx[i]), [], order]
        else:                                             # atom
            cap = int(tt.cap[i])
            order, nxt = next_atom_state(int(tt.bond[i]), cap, state)
            if order == 0:
                if state == 0:
                    f[1] = self._add_atom(cap)
                    self.counts[tt.elem[i]] += 1
                else:
                    self.broken = True                    # selfies would bond from an unplaced atom
            else:
                a = self._add_atom(cap)
                self.counts[tt.elem[i]] += 1
                src = f[1]
                self.bonds[(src, a)] = order
                self.bc[src] += order
                self.bc[a] += order
                f[1] = a
            f[0] = nxt
        self._close_exhausted()

    # -- queries ----------------------------------------------------------------------------- #

    def remaining(self):
        return None if self.target is None else self.target - self.counts

    def heavy_done(self):
        r = self.remaining()
        if r is None:
            return True
        r = r.copy()
        if self.tt.h_ix >= 0:
            r[self.tt.h_ix] = 0
        return self.missing == 0 and not r.any()

    def eos_legal(self):
        return (not self.broken and self.pending is None and len(self.caps) > 0 and self.heavy_done())

    def legal_mask(self):
        """numpy bool over the vocabulary."""
        tt = self.tt
        n = len(tt.kind)
        if self.broken:
            return np.zeros(n, bool)
        if self.pending is not None:
            return tt.is_index.copy()
        state = self.frames[-1][0]
        mask = np.zeros(n, bool)
        if state is not None:
            placed = tt.is_atom if state == 0 else tt.atom_cap_pos
            if self.target is not None:
                placed = placed & (self.remaining()[tt.elem_safe] > 0)
            mask |= placed
            if state > 1:
                mask |= tt.is_branch
            if state > 0:
                mask |= tt.is_ring
        if self.eos_legal():
            mask[tt.vocab.eos_id] = True
        return mask

    def unsat(self):
        """Rings + pi bonds of the molecule this prefix decodes to (selfies' _form_rings_bilocally)."""
        n = len(self.caps)
        if n == 0:
            return 0
        bc, bonds = self.bc[:], dict(self.bonds)
        for l, r, order in self.rings:
            if l == r:
                continue
            lfree, rfree = self.caps[l] - bc[l], self.caps[r] - bc[r]
            if lfree <= 0 or rfree <= 0:
                continue
            order = min(order, lfree, rfree)
            key = (min(l, r), max(l, r))
            if key in bonds:
                new = min(order + bonds[key], 3)
                bc[l] += new - bonds[key]
                bc[r] += new - bonds[key]
                bonds[key] = new
            else:
                bonds[key] = order
                bc[l] += order
                bc[r] += order
        return sum(bonds.values()) - n + 1

    def element_counts(self):
        return {e: int(c) for e, c in zip(self.tt.elements, self.counts) if c}


# --------------------------------------------------------------------------------------------- #
# Formulae
# --------------------------------------------------------------------------------------------- #

def parse_formula(formula):
    counts = {}
    for e, c in _FORMULA_RE.findall(formula):
        counts[e] = counts.get(e, 0) + (int(c) if c else 1)
    return counts


def dbe(counts):
    """Rings + pi bonds implied by a formula with standard valences (C4 N3 O2 X1 H1)."""
    x = sum(counts.get(e, 0) for e in ("F", "Cl", "Br", "I"))
    return 1 + counts.get("C", 0) - (counts.get("H", 0) + x) / 2 + counts.get("N", 0) / 2


def dbe_prune_applies(counts):
    return set(e for e, c in counts.items() if c) <= DBE_SAFE_ELEMENTS


# --------------------------------------------------------------------------------------------- #
# RDKit side: structure prep, ik14, reference tracker
# --------------------------------------------------------------------------------------------- #

def _chem():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def mol_formula(mol):
    """{element: count} incl. total H (implicit + explicit) of an RDKit mol."""
    counts = {}
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        counts[s] = counts.get(s, 0) + 1
        h = a.GetTotalNumHs()
        if h:
            counts["H"] = counts.get("H", 0) + h
    return counts


def ik14(smiles):
    Chem = _chem()
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    k = Chem.MolToInchiKey(m)
    return k[:14] if k else None


def prep_structure(smiles, elements=ALLOWED_ELEMENTS):
    """-> (status, canonical_smiles, selfies, formula). status 'ok' or the reason it is excluded:
    'parse', 'multi' (more than one component), 'element', 'encode' (selfies constraints)."""
    Chem = _chem()
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return "parse", None, None, None
    if len(Chem.GetMolFrags(m)) > 1:
        return "multi", None, None, None
    if any(a.GetSymbol() not in elements for a in m.GetAtoms()):
        return "element", None, None, None
    Chem.RemoveStereochemistry(m)
    for a in m.GetAtoms():
        a.SetIsotope(0)
    can = Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(can)
    try:
        s = sf.encoder(can, strict=True)
    except Exception:
        return "encode", can, None, None
    if s is None:
        return "encode", can, None, None
    return "ok", can, s, mol_formula(m)


def reference_state(tokens):
    """Slow truth for a token prefix: (element counts incl. explicit H atoms, rings + pi bonds)."""
    Chem = _chem()
    smi = sf.decoder("".join(tokens))
    if not smi:
        return {}, 0
    m = Chem.MolFromSmiles(smi, sanitize=False)
    counts = {}
    for a in m.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    order = sum(int(b.GetBondTypeAsDouble()) for b in m.GetBonds())
    n_comp = len(Chem.GetMolFrags(m, sanitizeFrags=False))
    return counts, order - m.GetNumAtoms() + n_comp


def tracker_parity(table, tokens):
    """Fast vs reference at every prefix. -> list of (prefix length, what, fast, reference)."""
    vocab = table.vocab
    t = Tracker(table)
    bad = []
    for k, tok in enumerate(tokens, 1):
        t = t.push(vocab.tok2id[tok])
        ref_counts, ref_u = reference_state(tokens[:k])
        if t.element_counts() != ref_counts:
            bad.append((k, "counts", t.element_counts(), ref_counts))
        if t.pending is None and t.unsat() != ref_u:
            bad.append((k, "unsat", t.unsat(), ref_u))
        if t.broken:
            bad.append((k, "broken", None, None))
    return bad


def walk_truth(table, tokens, formula, max_len=None, dbe_prune=False):
    """Mask soundness for one truth. -> list of (position, rule, token) violations.

    rules: 'count' (atom masked, element exhausted), 'grammar' (masked for any other reason),
    'eos' (</s> not legal after the last token), 'length' (> max_len), 'dbe' (unsat above DBE)."""
    vocab = table.vocab
    t = Tracker(table, formula)
    limit = dbe(formula) if dbe_prune and dbe_prune_applies(formula) else None
    bad = []
    for k, tok in enumerate(tokens):
        i = vocab.tok2id.get(tok, vocab.unk_id)
        if not t.legal_mask()[i]:
            r = t.remaining()
            count_rule = (table.kind[i] == K_ATOM and t.pending is None and r is not None
                          and r[table.elem_safe[i]] <= 0)
            bad.append((k, "count" if count_rule else "grammar", tok))
        t = t.push(i)
        if limit is not None and t.pending is None and t.unsat() > limit + 1e-9:
            bad.append((k, "dbe", tok))
    if not t.eos_legal():
        bad.append((len(tokens), "eos", None))
    if max_len is not None and len(tokens) > max_len:
        bad.append((len(tokens), "length", None))
    return bad


# --------------------------------------------------------------------------------------------- #
# P3: memory, decoder, batched constrained search (torch)
#
# Everything below is the P3 part (specs_novel/gate_report.md -> G3). The P0 part above stays
# torch-free; torch is imported here because every consumer of this part needs it anyway (Colab
# training, the Kaggle worker, the local tests).
# --------------------------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# must equal simsim_common.TEST_ADDUCTS -- test_novel_p3.check_adducts asserts it
TEST_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
CE_BINS = (15.0, 23.0, 36.0, 44.0)       # P0.2 quantiles; 5 intervals + "unknown" = 6 rows
N_CE_BINS = len(CE_BINS) + 2             # digitize gives 0..4, and 5 is "unknown"
CE_UNKNOWN = len(CE_BINS) + 1
FORMULA_SCALE = 10.0                     # counts are divided by this before the formula MLP


def ce_bin(ce, bins=CE_BINS):
    """Collision energy in eV -> bin index. None / NaN / <= 0 -> CE_UNKNOWN."""
    if ce is None:
        return CE_UNKNOWN
    try:
        v = float(ce)
    except (TypeError, ValueError):
        return CE_UNKNOWN
    if not np.isfinite(v) or v <= 0.0:
        return CE_UNKNOWN
    return int(np.digitize(v, bins))


def formula_vector(counts, elements=ALLOWED_ELEMENTS):
    """{element: count} -> float32 vector over `elements`, hydrogen INCLUDED.

    Deliberately not TokenTable.counts_vector: that one drops H (the vocabulary has no free H token),
    but the formula token conditions on the full formula, H and all."""
    v = np.zeros(len(elements), np.float32)
    ix = {e: i for i, e in enumerate(elements)}
    for e, c in counts.items():
        if e in ix:
            v[ix[e]] = c
    return v


class DreamsSelfiesDecoder(nn.Module):
    """SELFIES decoder over frozen DreaMS embeddings (plan section 3.5).

    memory = [W_e.emb_i + E_adduct[a_i] + E_ce[c_i]]_i  (+)  [MLP(formula_counts)]

    There is no positional encoding on the memory: the spectra of a molecule are a set, and
    test_novel_p3 asserts the permutation invariance. Input and output token embeddings are tied.
    """

    def __init__(self, n_vocab, *, pad_id, sos_id, eos_id, d_model=512, n_layers=8, n_heads=8,
                 d_ff=2048, dropout=0.1, max_len=171, n_elements=len(ALLOWED_ELEMENTS),
                 n_adducts=len(TEST_ADDUCTS), n_ce_bins=N_CE_BINS, d_emb=1024):
        super().__init__()
        self.hparams = dict(n_vocab=n_vocab, pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                            d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                            dropout=dropout, max_len=max_len, n_elements=n_elements,
                            n_adducts=n_adducts, n_ce_bins=n_ce_bins, d_emb=d_emb)
        self.pad_id, self.sos_id, self.eos_id = pad_id, sos_id, eos_id
        self.d_model, self.max_len = d_model, max_len

        # -- memory side --
        self.emb_proj = nn.Linear(d_emb, d_model)
        self.adduct_emb = nn.Embedding(n_adducts, d_model)
        self.ce_emb = nn.Embedding(n_ce_bins, d_model)
        self.formula_mlp = nn.Sequential(nn.Linear(n_elements, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
        self.mem_norm = nn.LayerNorm(d_model)

        # -- token side --
        self.tok_emb = nn.Embedding(n_vocab, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len + 2, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        self.out_bias = nn.Parameter(torch.zeros(n_vocab))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[self.pad_id].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    # -- memory ------------------------------------------------------------------------------- #

    def build_memory(self, embs, adduct_ids, ce_ids, formula_counts, spec_mask=None):
        """embs (B, K, d_emb) float, adduct_ids / ce_ids (B, K) long, formula_counts (B, E) float,
        spec_mask (B, K) bool with True = a real spectrum.

        -> memory (B, K+1, D), mem_pad (B, K+1) bool with True = ignore (nn.Transformer convention).
        The formula token is last and is never padded."""
        if embs.dim() == 2:                       # a single molecule, (K, d_emb)
            embs, adduct_ids, ce_ids = embs[None], adduct_ids[None], ce_ids[None]
            formula_counts = formula_counts[None]
            spec_mask = None if spec_mask is None else spec_mask[None]
        b, k = embs.shape[0], embs.shape[1]
        if spec_mask is None:
            spec_mask = torch.ones(b, k, dtype=torch.bool, device=embs.device)
        m_spec = self.emb_proj(embs) + self.adduct_emb(adduct_ids) + self.ce_emb(ce_ids)
        m_form = self.formula_mlp(formula_counts / FORMULA_SCALE).unsqueeze(1)
        memory = self.mem_norm(torch.cat([m_spec, m_form], dim=1))
        mem_pad = torch.cat([~spec_mask, torch.zeros(b, 1, dtype=torch.bool, device=embs.device)],
                            dim=1)
        # zero the padded spectrum tokens so they cannot leak through a fully-masked attention row
        memory = memory.masked_fill(mem_pad.unsqueeze(-1), 0.0)
        return memory, mem_pad

    # -- decoding ----------------------------------------------------------------------------- #

    def forward(self, tok_ids, memory, mem_pad=None):
        """tok_ids (B, L) long, already prefixed with <s>. -> logits (B, L, V)."""
        b, l = tok_ids.shape
        assert l <= self.max_len + 2, f"sequence length {l} exceeds max_len + 2"
        pos = torch.arange(l, device=tok_ids.device)
        x = self.drop(self.tok_emb(tok_ids) * math.sqrt(self.d_model) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(l, l, dtype=torch.bool, device=tok_ids.device), diagonal=1)
        h = self.dec(x, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return h @ self.tok_emb.weight.T + self.out_bias

    def next_logits(self, tok_ids, memory, mem_pad=None):
        """Logits for the token after the last position of `tok_ids`. (B, V).

        Recomputes the whole prefix each call: no KV cache. At this model size that is the simpler
        correct thing; a cache is a P4 optimisation (search_plan.md v1.1 section 0.5)."""
        return self.forward(tok_ids, memory, mem_pad)[:, -1, :]


# --------------------------------------------------------------------------------------------- #
# Search: shared scaffolding
# --------------------------------------------------------------------------------------------- #

NEG_INF = -1e9


def _expand_memory(memory, mem_pad, n):
    """A single molecule's memory -> n identical rows, so a whole beam is one forward pass."""
    if memory.shape[0] == n:
        return memory, mem_pad
    assert memory.shape[0] == 1, f"expected a single-molecule memory, got {memory.shape[0]}"
    pad = None if mem_pad is None else mem_pad.expand(n, -1)
    return memory.expand(n, -1, -1), pad


def _masks_to_tensor(trackers, device):
    return torch.from_numpy(np.stack([t.legal_mask() for t in trackers])).to(device)


def _logprobs(model, tok_ids, memory, mem_pad, trackers):
    """Masked log-probabilities of the next token. Illegal tokens get NEG_INF, so a beam can never
    step outside the formula. Rows with no legal token at all are reported as `dead`."""
    logits = model.next_logits(tok_ids, memory, mem_pad).float()
    legal = _masks_to_tensor(trackers, logits.device)
    dead = ~legal.any(dim=1)
    logits = logits.masked_fill(~legal, NEG_INF)
    return torch.log_softmax(logits, dim=-1), legal, dead


def _leaf(vocab, ids, target_counts):
    """A finished token sequence -> (smiles, ik14), or None if it fails the plan section 3.6 filter.

    The tracker already guarantees the heavy-atom counts; what is checked here is everything it
    cannot know: that the string decodes and sanitises, that it is one component, and that the
    FULL formula including hydrogen matches. Hydrogen is only fixed once SELFIES has filled the
    valences, which is why this check does real work."""
    Chem = _chem()
    smi = sf.decoder(vocab.decode(ids))
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None or len(Chem.GetMolFrags(m)) > 1:
        return None
    if mol_formula(m) != target_counts:
        return None
    can = Chem.MolToSmiles(m)
    k = Chem.MolToInchiKey(m)
    return (can, k[:14]) if k else None


FUNNEL_KEYS = ("sequences", "eos", "valid", "unique", "dead", "truncated")


def new_funnel():
    """The G3 funnel, counted by the searches themselves (gate_report.md -> G3, recorded-not-gating).

    sequences  rollouts / beams started
    eos        sequences that reached </s>, i.e. the heavy-atom formula was satisfied exactly
    valid      of those, the ones that decode, sanitise, are one component and match the FULL
               formula including hydrogen
    unique     distinct ik14 among the valid ones
    dead       sequences abandoned because no token was legal (valence ran out)
    truncated  sequences still unfinished at MAX_LEN
    """
    return dict.fromkeys(FUNNEL_KEYS, 0)


def _collect(out, vocab, ids, logp, target_counts, fn=None):
    """Add a finished sequence to `out`, an ik14 -> (smiles, logp) dict keeping the best logp."""
    if fn is not None:
        fn["eos"] += 1
    leaf = _leaf(vocab, ids, target_counts)
    if leaf is None:
        return False
    smi, key = leaf
    if fn is not None:
        fn["valid"] += 1
    prev = out.get(key)
    if prev is None or logp > prev[1]:
        out[key] = (smi, logp)
    return True


def _prep_search(model, memory, mem_pad, formula, table):
    """Shared entry: canonical target counts, the root tracker, and the <s> row."""
    counts = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    counts = {e: int(c) for e, c in counts.items() if c}
    root = Tracker(table, counts)
    assert root.missing == 0, f"formula {formula} has elements outside the vocabulary"
    device = memory.device
    sos = torch.full((1, 1), model.sos_id, dtype=torch.long, device=device)
    return counts, root, sos


# --------------------------------------------------------------------------------------------- #
# Arm A: prefix-seeded batched sampling  (search_plan.md v1.1 Method 1, corrected)
# --------------------------------------------------------------------------------------------- #

def _seed_prefixes(model, memory, mem_pad, table, root, sos, l_seed, k_roots, width, prefix_cap):
    """Beam-expand to depth `l_seed`, then keep `k_roots` prefixes with a spread of scaffolds.

    Seeding at depth 1 would be pointless: at derivation state 0 the mask admits only atom tokens
    ([Branch*] needs state > 1, [Ring*] needs state > 0) and next_atom_state returns order 0 for
    every bond prefix, so [C] / [=C] / [#C] are degenerate. The distinct roots at depth 1 are just
    the elements of the formula. Ring and branch structure -- the scaffold -- first appears a few
    tokens in, which is why the default is l_seed = 4.

    `prefix_cap` bounds how many kept prefixes may share their first two tokens, so the K roots
    cannot all collapse into one lineage. The cap is relaxed if too few prefixes survive it."""
    beam = [(sos, root, 0.0)]                      # (ids (1, L), tracker, logp)
    for _ in range(l_seed):
        ids = torch.cat([b[0] for b in beam], dim=0)
        trackers = [b[1] for b in beam]
        mem, pad = _expand_memory(memory, mem_pad, len(beam))
        lp, legal, dead = _logprobs(model, ids, mem, pad, trackers)
        cand = []
        for i, (bi, t, base) in enumerate(beam):
            if dead[i]:
                continue
            for tok in torch.nonzero(legal[i], as_tuple=False).flatten().tolist():
                if tok == model.eos_id:            # a complete molecule this early is not a seed
                    continue
                cand.append((base + float(lp[i, tok]), i, tok))
        if not cand:
            break
        cand.sort(key=lambda c: -c[0])
        nxt = []
        for score, i, tok in cand[:width]:
            bi, t, _ = beam[i]
            step = torch.tensor([[tok]], dtype=torch.long, device=bi.device)
            nxt.append((torch.cat([bi, step], dim=1), t.push(tok), score))
        beam = nxt
    beam.sort(key=lambda b: -b[2])
    for cap in (prefix_cap, len(beam)):            # relax the spread rule rather than return fewer
        seen, kept = {}, []
        for b in beam:
            sig = tuple(b[0][0, 1:3].tolist())
            if seen.get(sig, 0) >= cap:
                continue
            seen[sig] = seen.get(sig, 0) + 1
            kept.append(b)
            if len(kept) == k_roots:
                break
        if len(kept) >= min(k_roots, len(beam)):
            return kept
    return kept


def _nucleus(lp, top_p):
    """Zero out the tail of each row beyond cumulative probability `top_p` (log space in, log out)."""
    srt, idx = torch.sort(lp, dim=-1, descending=True)
    keep = torch.cumsum(srt.exp(), dim=-1) - srt.exp() < top_p
    keep[:, 0] = True                              # never drop the mode
    drop = torch.zeros_like(keep).scatter(1, idx, ~keep)
    return torch.log_softmax(lp.masked_fill(drop, NEG_INF), dim=-1)


@torch.no_grad()
def search_prefix_seeded(model, memory, formula, table, *, mem_pad=None, l_seed=4, k_roots=8,
                         m_rollout=32, temperature=0.7, top_p=0.90, max_len=None, seed=None,
                         stats=None):
    """P3 bake-off arm A. -> [(smiles, logp)], ik14-deduped, best logp per ik14.

    Pass `stats` as a dict to have the funnel counts written into it (see new_funnel)."""
    model.eval()
    max_len = max_len or model.max_len
    counts, root, sos = _prep_search(model, memory, mem_pad, formula, table)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=memory.device).manual_seed(int(seed))
    prefixes = _seed_prefixes(model, memory, mem_pad, table, root, sos, l_seed, k_roots,
                              width=max(k_roots * 8, 32), prefix_cap=2)
    out = {}
    fn = stats if stats is not None else None
    if fn is not None:
        fn.update({k: fn.get(k, 0) for k in FUNNEL_KEYS})
        fn["roots"] = fn.get("roots", 0) + len(prefixes)
    if not prefixes:
        return []
    if fn is not None:
        fn["sequences"] += len(prefixes) * m_rollout
    ids = torch.cat([p[0].repeat(m_rollout, 1) for p in prefixes], dim=0)
    trackers = [p[1]._copy() for p in prefixes for _ in range(m_rollout)]
    logp = [p[2] for p in prefixes for _ in range(m_rollout)]
    alive = list(range(ids.shape[0]))
    while alive and ids.shape[1] < max_len + 1:
        act = [trackers[i] for i in alive]
        mem, pad = _expand_memory(memory, mem_pad, len(alive))
        lp, legal, dead = _logprobs(model, ids[alive], mem, pad, act)
        lp = _nucleus(lp / max(temperature, 1e-6), top_p)
        tok = torch.multinomial(lp.exp(), 1, generator=gen).flatten()
        step = torch.full((ids.shape[0], 1), model.pad_id, dtype=torch.long, device=ids.device)
        step[alive] = tok[:, None]
        ids = torch.cat([ids, step], dim=1)
        nxt = []
        for j, i in enumerate(alive):
            if dead[j]:
                if fn is not None:
                    fn["dead"] += 1
                continue
            t = int(tok[j])
            logp[i] += float(lp[j, t])
            if t == model.eos_id:
                _collect(out, table.vocab, ids[i, 1:-1].tolist(), logp[i], counts, fn)
                continue
            trackers[i] = trackers[i].push(t)
            nxt.append(i)
        alive = nxt
    if fn is not None:
        fn["truncated"] += len(alive)
        fn["unique"] += len(out)
    return sorted(((s, lg) for s, lg in out.values()), key=lambda c: -c[1])


# --------------------------------------------------------------------------------------------- #
# Arms B and C: masked beam search, plain and cross-beam diversified
# --------------------------------------------------------------------------------------------- #

@torch.no_grad()
def search_masked_beam(model, memory, formula, table, *, mem_pad=None, beam=256, max_len=None,
                       alpha=0.0, max_leaves=None, stats=None):
    """P3 bake-off arm B (alpha = 0) and arm C (alpha > 0). -> [(smiles, logp)], ik14-deduped.

    With alpha > 0 the score used for *selection* is penalised by how often each ring / branch token
    was taken across the previous step's beams, which spreads the beams over scaffold topologies
    instead of letting them collapse onto one core. The returned logp is always the UNPENALISED
    cumulative log-probability, because P4 orders novel candidates on it.

    This is the non-circular reading of search_plan.md Method 3: the penalty uses the previous
    step's counts, since a penalty defined by the current step's own choices is not well posed."""
    model.eval()
    max_len = max_len or model.max_len
    max_leaves = beam * 4 if max_leaves is None else max_leaves
    counts, root, sos = _prep_search(model, memory, mem_pad, formula, table)
    scaffold = torch.from_numpy(table.is_branch | table.is_ring).to(memory.device)
    live = [(sos, root, 0.0)]                      # (ids, tracker, unpenalised logp)
    out = {}
    fn = stats if stats is not None else None
    if fn is not None:
        fn.update({k: fn.get(k, 0) for k in FUNNEL_KEYS})
    prev = torch.zeros(len(table.kind), device=memory.device)
    for _ in range(max_len + 1):
        if not live:
            break
        ids = torch.cat([b[0] for b in live], dim=0)
        trackers = [b[1] for b in live]
        mem, pad = _expand_memory(memory, mem_pad, len(live))
        lp, legal, dead = _logprobs(model, ids, mem, pad, trackers)
        pen = (alpha * (prev / max(len(live), 1)) * scaffold)[None, :] if alpha else 0.0
        cand = []
        for i, (bi, t, base) in enumerate(live):
            if dead[i]:
                if fn is not None:
                    fn["dead"] += 1
                continue
            for tok in torch.nonzero(legal[i], as_tuple=False).flatten().tolist():
                real = base + float(lp[i, tok])
                rank = real - (float(pen[0, tok]) if alpha else 0.0)
                cand.append((rank, real, i, tok))
        if not cand:
            break
        cand.sort(key=lambda c: -c[0])
        nxt, taken = [], torch.zeros_like(prev)
        for _rank, real, i, tok in cand:
            full = len(nxt) >= beam
            if full and len(out) >= max_leaves:
                break
            bi, t, _ = live[i]
            if tok == model.eos_id:
                if len(out) < max_leaves:
                    _collect(out, table.vocab, bi[0, 1:].tolist(), real, counts, fn)
                continue
            if full:
                continue
            step = torch.tensor([[tok]], dtype=torch.long, device=bi.device)
            nxt.append((torch.cat([bi, step], dim=1), t.push(tok), real))
            if fn is not None:
                fn["sequences"] += 1
            taken[tok] += 1
        live, prev = nxt, taken
    if fn is not None:
        fn["truncated"] += len(live)
        fn["unique"] += len(out)
    return sorted(((s, lg) for s, lg in out.values()), key=lambda c: -c[1])


def search_diverse_beam(model, memory, formula, table, *, mem_pad=None, beam=256, alpha=0.6,
                        max_len=None, max_leaves=None, stats=None):
    """P3 bake-off arm C: search_masked_beam with the cross-beam penalty switched on."""
    return search_masked_beam(model, memory, formula, table, mem_pad=mem_pad, beam=beam,
                              max_len=max_len, alpha=alpha, max_leaves=max_leaves, stats=stats)


SEARCHES = {"prefix_seeded": search_prefix_seeded,
            "masked_beam": search_masked_beam,
            "diverse_beam": search_diverse_beam}


# --------------------------------------------------------------------------------------------- #
# Selecting the 25 that are scored
# --------------------------------------------------------------------------------------------- #

def morgan_fps(smiles, radius=2, n_bits=2048):
    """MassSpecGym's fingerprint definition -- Morgan, radius 2, 2048 bits (`utils.morgan_fp`).

    Their code calls `Chem.GetMorganFingerprintAsBitVect`, which does not exist on RDKit 2026.03.3
    (the metric's pinned version): it moved to AllChem and is now deprecated in favour of
    MorganGenerator. So the *definition* is reused, not the call. The generator returns bit-identical
    fingerprints -- test_novel_p3.check_morgan asserts that against the deprecated function wherever
    it is still reachable."""
    Chem = _chem()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        fps.append(None if m is None else gen.GetFingerprint(m))
    return fps, DataStructs


def max_tanimoto(cands, truth_smiles):
    """The metric behind G3: the best Tanimoto to the truth over a candidate list. 0.0 if empty."""
    if not cands:
        return 0.0
    fps, DataStructs = morgan_fps([truth_smiles] + [c[0] for c in cands])
    if fps[0] is None:
        return 0.0
    ok = [f for f in fps[1:] if f is not None]
    if not ok:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(fps[0], ok)))


def select_top25(cands, mode="maxmin", k=25):
    """Cut a candidate list to `k`. `cands` is [(smiles, logp)], already ik14-deduped.

    "logp"   -- the most likely k. The honest decoder-likelihood ranking.
    "maxmin" -- start from the most likely candidate, then repeatedly add whichever candidate is
                furthest (in Tanimoto) from everything already chosen. G3 scores the *maximum*
                similarity over the k, so covering chemical space beats stacking near-duplicates
                of one mode (search_plan.md v1.1 section 0.2).
    """
    if mode not in ("logp", "maxmin"):
        raise ValueError(f"unknown selection mode {mode!r}")
    ranked = sorted(cands, key=lambda c: -c[1])
    if mode == "logp" or len(ranked) <= k:
        return ranked[:k]
    fps, DataStructs = morgan_fps([c[0] for c in ranked])
    pool = [i for i, f in enumerate(fps) if f is not None]
    if len(pool) <= k:
        return [ranked[i] for i in pool] or ranked[:k]
    chosen = [pool[0]]                             # pool is logp-sorted, so this is the mode
    rest = pool[1:]
    best = list(DataStructs.BulkTanimotoSimilarity(fps[chosen[0]], [fps[i] for i in rest]))
    while len(chosen) < k and rest:
        j = min(range(len(rest)), key=lambda x: (best[x], -ranked[rest[x]][1]))
        chosen.append(rest.pop(j))
        best.pop(j)
        if not rest:
            break
        sims = DataStructs.BulkTanimotoSimilarity(fps[chosen[-1]], [fps[i] for i in rest])
        best = [max(b, s) for b, s in zip(best, sims)]
    return [ranked[i] for i in chosen]
