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

    # --- ICEBERG ----------------------------------------------------------------------------------------
    use_sim: bool = True
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
