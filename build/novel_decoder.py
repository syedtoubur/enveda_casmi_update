"""specs_novel single source of truth for the SELFIES side (plan §4, §5.5). P0 part: no model yet.

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

Needs numpy + selfies (pinned); RDKit only for reference_state / prep_structure / ik14.
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
