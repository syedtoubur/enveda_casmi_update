"""Every setting of the pipeline in one dataclass (replaces the notebooks' global CFG)."""
from dataclasses import asdict, dataclass, replace

COVERED_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M+Na]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Cl]-")   # Track 3 G2b


@dataclass(frozen=True)
class Config:
    # --- candidate window (phase1 values) ---------------------------------------------------------------
    ppm_win: float = 8.5
    ppm_center: float = 1.45            # timsTOF precursor reads +1.45 ppm high
    ppm_expand: tuple = (8.5, 20.0, 45.0, 80.0)
    ppm_fallback: float = 30.0
    backfill_ppm: float = 120.0
    expand_backfill: bool = True
    topn: int = 25

    # --- candidate pool (R1) ------------------------------------------------------------------------------
    pool_source: str = "r0"             # "r0": COCONUT + train built at runtime; "r1": casmi26-scratch-pool dataset
    pool_drop_e180: bool = False        # R1b: drop structures whose only provenance is enveda-180

    # --- spectrum cleaning: ONE max_peaks for library sim, MetFrag-lite and ICEBERG sim -------------------
    int_floor: float = 0.002
    max_peaks: int = 100
    mz_tol: float = 0.01
    int_power: float = 1.0
    ent_weight: bool = True
    prec_margin: float = 2.0            # predicted peaks above precursor + 2 Da are dropped

    # --- library / MetFrag-lite -------------------------------------------------------------------------
    struct_mass: bool = True            # library neutral mass from the label formula
    max_bonds: int = 60
    frag_workers: int = 4

    # --- two-tier ranking ---------------------------------------------------------------------------------
    tau: float = 0.75

    # --- R2 class vote (NPClassifier superclass of analog library hits -> tier-2 bonus) -----------------------
    class_vote: bool = False
    vote_beta: float = 0.10             # pre-registered, never tuned
    vote_k: int = 10
    vote_min_sim: float = 0.5
    vote_ppm: float = 10.0              # direct + NP delta-list windows

    # --- R3 FPNet channel (f.z fused with sv by RRF inside tier 2, stages A + B) -------------------------------
    fp_channel: str = ""                # "" (off) | "own" (our FPNet export, family P) | "ahmed" (fpnet_full1, family A)
    fp_rrf_k: float = 3.0               # pre-registered, never tuned
    # --- R4 FPNet family fusion: one f.z channel per bank, each its own RRF term (overrides fp_channel when set) ---
    fp_banks: tuple = ()                # e.g. ("ahmed", "prv") = full1 (family A) + prvsiyan casmi26-fp-models-v2 (P)
    fp_weights: tuple = ()              # RRF weight per bank (default 1.0 each); R4: (1.0, 0.6), never tuned
    fp_stagec: bool = False             # R4: order the stage-C mass-nearest top-up by the fused FP key

    def fp_kinds(self):
        """The FPNet banks in use: fp_banks, else (fp_channel,) when set, else ()."""
        return tuple(self.fp_banks) or ((self.fp_channel,) if self.fp_channel else ())

    def fp_w(self):
        k = self.fp_kinds()
        w = tuple(float(x) for x in self.fp_weights) or (1.0,) * len(k)
        if len(w) != len(k):
            raise ValueError(f"fp_weights {self.fp_weights} do not match fp banks {k}")
        return w

    # --- R5 analogue channel (open +-200 Da analogues -> PSSM or Tanimoto kernel; a tier-2 RRF list) ----------
    analog_channel: str = ""            # "" (off) | "pssm" | "tani"; specs/specs_scratch/pssm.md
    analog_open_da: float = 200.0
    analog_smin: float = 0.5
    analog_m: int = 100                 # PSSM: top analogues (deduplicated by structure)
    analog_k: int = 20                  # Tanimoto kernel: top analogues
    analog_alpha: float = 1.0           # pseudocount: background analogues in the profile
    analog_clip: float = 3.0            # |W_j| cap
    analog_w: float = 1.0               # RRF weight, pre-registered, never tuned
    analog_min_n: int = 3               # fewer analogues -> no list for that molecule
    analog_q_n: int = 50_000            # pool sample for the background q when no analog_q.npy is attached
    analog_exclude_train: bool = True   # D4: F_TRAIN candidates get the neutral term in the list; False = score all

    # --- simulator (ICEBERG, or GLACIER for the R1 GLACIER twins) -------------------------------------------
    use_sim: bool = True
    sim_model: str = "iceberg"          # "iceberg" | "glacier" (casmi26-glacier: [M+H]+ / [M+Na]+ only)
    gl_batch: int = 32                  # GLACIER max batch (the runner also caps B * N_atoms^2 <= bs * 1600)
    gl_batch_retry: int = 8
    sim_tier1: bool = False
    merge_rule: str = "max"
    covered: tuple = COVERED_ADDUCTS
    ce_mode: str = "ev"
    instrument: str = "QTOF"
    ce_default: float = 40.0
    batch: int = 128
    batch_retry: int = 32
    chunk: int = 20_000
    sparse_k: int = 100
    rate: float = 36.6                  # requests/s on one T4 (replaced by the measured rate)
    startup_s: float = 10.4
    hard_budget_h: float = 7.5
    pass2_s_per_mol: float = 1.0
    min_ok_fast: float = 0.99

    # --- run mode -----------------------------------------------------------------------------------------
    allow_fast: bool = True
    fast_dev_molecules: int = 20
    public_n_mol: int = 400
    public_n_spec: int = 1213
    force_full: bool = False

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)
