"""Adduct physics and formula masses (no RDKit)."""
import re

import numpy as np

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
ELECTRON = 0.00054857990
PROTON = MASS["H"] - ELECTRON
H2O = 2 * MASS["H"] + MASS["O"]
NH4 = MASS["N"] + 4 * MASS["H"]
FORMATE = MASS["C"] + 2 * MASS["H"] + 2 * MASS["O"]
ACETATE = 2 * MASS["C"] + 4 * MASS["H"] + 2 * MASS["O"]

# adduct -> (n molecules, charge, mass delta): precursor m/z = (n * M + delta) / z
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - ELECTRON), "[M+Na]+": (1, 1, MASS["Na"] - ELECTRON),
    "[M+K]+": (1, 1, MASS["K"] - ELECTRON), "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O), "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -ELECTRON),
    "[M-H2O]+": (1, 1, -ELECTRON - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS["C"] + 4 * MASS["H"] + MASS["O"]),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS["C"] + 3 * MASS["H"] + MASS["N"]),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS["Cl"] + ELECTRON), "[M]-": (1, 1, ELECTRON),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS["Na"] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS["Na"] - ELECTRON), "[2M+NH4]+": (2, 1, NH4 - ELECTRON),
    "[2M+K]+": (2, 1, MASS["K"] - ELECTRON), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS["Na"] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}


def neutral_mass(mz, adduct):
    """Vectorised precursor m/z -> neutral mass; NaN for an unknown adduct."""
    mz = np.asarray(mz, np.float64)
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = ad == a
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


def precursor_of(mass, adduct):
    """Neutral exact mass -> theoretical precursor m/z (singly charged, one molecule adducts)."""
    n, z, d = ADDUCTS[adduct]
    return (np.asarray(mass, np.float64) * n + d) / z


_MONO = {"C": 12.0, "H": 1.0078250319, "N": 14.0030740052, "O": 15.9949146221, "S": 31.97207069,
         "P": 30.97376151, "F": 18.99840320, "Cl": 34.96885271, "Br": 78.9183376, "I": 126.904468,
         "Si": 27.9769265327, "Se": 79.9165196, "B": 11.0093055, "Na": 22.98976928, "K": 38.96370649,
         "Fe": 55.9349375, "D": 2.0141017778, "As": 74.9215965, "Li": 7.0160034, "Sn": 119.9021947}
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(f):
    """Monoisotopic mass of a molecular formula string; NaN if unparseable or an exotic element."""
    if not isinstance(f, str):
        return np.nan
    s, pos = 0.0, 0
    for m in _TOK.finditer(f):
        if m.start() != pos or m.group(1) not in _MONO:
            return np.nan
        pos = m.end()
        s += _MONO[m.group(1)] * (int(m.group(2)) if m.group(2) else 1)
    return s if pos == len(f) else np.nan


def query_target(nm, ppm_center):
    """A molecule's neutral masses -> centred target mass (median, corrected for the timsTOF offset)."""
    nm = np.asarray(nm, np.float64)
    nm = nm[np.isfinite(nm)]
    return float(np.median(nm)) / (1.0 + ppm_center / 1e6) if len(nm) else np.nan
