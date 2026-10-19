"""Biogenetic Knapsack Vector Mutator.

Solves multi-step precursor mass deficits (delta_M +/- 10 ppm) using biochemical
additions and chops up to depth 3, applies transformations to seed analogs,
and deduplicates candidates on InChIKey14.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey

# Accurate monoisotopic atomic masses
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100,
            P=30.97376163, F=18.99840322, Cl=34.96885268, Br=78.9183371)

def calc_exact_delta(d: Dict[str, int]) -> float:
    return float(sum(MASS[k] * v for k, v in d.items()))

# Biological Basis Transformations (SMARTS, Formula Delta, Description)
BASIS_TRANSFORMS = {
    # 1. Hydroxylation / Oxygenation (+O, +15.994915 Da)
    'hydroxyl_arom': ('[c;H1:1]>>[c:1]O', dict(O=1)),
    'hydroxyl_aliph': ('[C;H2,H3;!$(C=O):1]>>[C:1]O', dict(O=1)),
    'hydroxyl_ch': ('[C;H1;!$(C=O):1]>>[C:1]O', dict(O=1)),
    
    # 2. Methylation (+CH2, +14.015650 Da)
    'methyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C', dict(C=1, H=2)),
    'methyl_N': ('[N;H1,H2:1]>>[N:1]C', dict(C=1, H=2)),
    'methyl_c': ('[c;H1:1]>>[c:1]C', dict(C=1, H=2)),
    
    # 3. Methoxylation (+CH2O, +30.010565 Da)
    'methoxy_arom': ('[c;H1:1]>>[c:1]OC', dict(C=1, H=2, O=1)),
    
    # 4. Acetylation (+C2H2O, +42.010565 Da)
    'acetyl_O': ('[O;H1;!$(OC=O):1]>>[O:1]C(C)=O', dict(C=2, H=2, O=1)),
    'acetyl_N': ('[N;H1,H2;!$(NC=O):1]>>[N:1]C(C)=O', dict(C=2, H=2, O=1)),
    
    # 5. Glycosylation / Glycans (+C6H10O5 hexose +162.052823 Da, deoxyhexose +146.057909 Da, pentose +132.042259 Da)
    'hexose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OC(CO)C(O)C(O)C1O', dict(C=6, H=10, O=5)),
    'deoxyhexose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OC(C)C(O)C(O)C1O', dict(C=6, H=10, O=4)),
    'pentose_O': ('[O;H1;!$(OC=O):1]>>[O:1]C1OCC(O)C(O)C1O', dict(C=5, H=8, O=4)),
    'hexose_C': ('[c;H1:1]>>[c:1]C1OC(CO)C(O)C(O)C1O', dict(C=6, H=10, O=5)),
    
    # 6. Hydrogenation (+H2, +2.015650 Da) & Dehydrogenation (-H2, -2.015650 Da)
    'hydrogenate_CC': ('[C:1]=[C:2]>>[C:1][C:2]', dict(H=2)),
    'dehydrogenate': ('[C;H1,H2:1][C;H1,H2:2]>>[C:1]=[C:2]', dict(H=-2)),
    'oxidize_CHOH': ('[C;H1:1][O;H1:2]>>[C:1]=[O:2]', dict(H=-2)),
    
    # 7. Chops / Removals
    'demethyl_O': ('[O:1][CH3]>>[O:1]', dict(C=-1, H=-2)),
    'demethyl_N': ('[N:1][CH3]>>[N:1]', dict(C=-1, H=-2)),
    'dehydroxyl_arom': ('[c:1][OH]>>[c:1]', dict(O=-1)),
    'dehydroxyl_aliph': ('[C;!$(C=O):1][OH]>>[C:1]', dict(O=-1)),
    'deacetyl_O': ('[O:1]C(=O)[CH3]>>[O:1]', dict(C=-2, H=-2, O=-1)),
    'demethoxy': ('[c:1]O[CH3]>>[c:1]', dict(C=-1, H=-2, O=-1)),
    'dehexose': ('[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]', dict(C=-6, H=-10, O=-5)),
    'decarboxyl': ('[C,c:1]C(=O)[OH]>>[C,c:1]', dict(C=-1, O=-2)),
}

# Compile exact delta masses
TRANSFORM_DELTAS = {name: calc_exact_delta(d) for name, (smarts, d) in BASIS_TRANSFORMS.items()}
_RXN_CACHE = {}

def get_rxn(name: str):
    if name not in _RXN_CACHE:
        _RXN_CACHE[name] = AllChem.ReactionFromSmarts(BASIS_TRANSFORMS[name][0])
    return _RXN_CACHE[name]

class KnapsackSolver:
    """Solves integer combinations of transforms up to depth 3 that match delta_M."""
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.combos = []       # list of (tuple of names, total_delta)
        self.delta_arr = None  # sorted np.ndarray for rapid searchsorted
        self._build_combos()

    def _build_combos(self):
        names = list(BASIS_TRANSFORMS.keys())
        combos = []
        # Depth 1
        for n in names:
            combos.append(((n,), TRANSFORM_DELTAS[n]))
        # Depth 2
        if self.max_depth >= 2:
            for a, b in itertools.combinations_with_replacement(names, 2):
                combos.append(((a, b), TRANSFORM_DELTAS[a] + TRANSFORM_DELTAS[b]))
        # Depth 3
        if self.max_depth >= 3:
            for a, b, c in itertools.combinations_with_replacement(names, 3):
                combos.append(((a, b, c), TRANSFORM_DELTAS[a] + TRANSFORM_DELTAS[b] + TRANSFORM_DELTAS[c]))
        
        # Sort by total mass delta
        combos.sort(key=lambda x: x[1])
        self.combos = combos
        self.delta_arr = np.array([c[1] for c in combos], dtype=np.float64)

    def find_combos(self, target_mass: float, seed_mass: float, ppm: float = 10.0) -> List[Tuple[Tuple[str, ...], float]]:
        delta_m = target_mass - seed_mass
        tol = max(0.003, target_mass * ppm * 1e-6)
        low = delta_m - tol
        high = delta_m + tol
        
        idx_low = np.searchsorted(self.delta_arr, low, side='left')
        idx_high = np.searchsorted(self.delta_arr, high, side='right')
        
        matching = []
        for i in range(idx_low, idx_high):
            matching.append(self.combos[i])
        return matching

def get_mol_exact_mass(m: Chem.Mol) -> float:
    from rdkit.Chem import Descriptors
    return Descriptors.ExactMolWt(m)

def apply_transform_to_mol(mol: Chem.Mol, name: str, max_prods: int = 32) -> List[Chem.Mol]:
    rxn = get_rxn(name)
    out = []
    try:
        prods = rxn.RunReactants((mol,))
    except Exception:
        return out
    seen_smi = set()
    for p in prods:
        m = p[0]
        try:
            Chem.SanitizeMol(m)
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if smi in seen_smi:
            continue
        seen_smi.add(smi)
        out.append(m)
        if len(out) >= max_prods:
            break
    return out

def mutate_seed(seed_smiles: str, target_mass: float, solver: KnapsackSolver,
                max_mutants: int = 100, ppm: float = 10.0) -> List[Dict]:
    """Mutate a seed analog to target mass via knapsack combinations."""
    mol = Chem.MolFromSmiles(seed_smiles)
    if mol is None:
        return []
    
    seed_mass = get_mol_exact_mass(mol)
    combos = solver.find_combos(target_mass, seed_mass, ppm=ppm)
    if not combos:
        return []
    
    # Sort combos by simplicity (fewer steps first)
    combos.sort(key=lambda x: len(x[0]))
    
    tol = max(0.003, target_mass * ppm * 1e-6)
    out_mutants = []
    seen_ik14 = set()
    budget = 400  # Reaction application budget per seed
    
    for combo, theoretical_delta in combos:
        cur_mols = [mol]
        for depth, rxn_name in enumerate(combo):
            next_mols = []
            for m in cur_mols[:16]:
                next_mols.extend(apply_transform_to_mol(m, rxn_name, max_prods=16 if depth == 0 else 8))
                budget -= 1
                if budget <= 0:
                    break
            cur_mols = next_mols
            if not cur_mols or budget <= 0:
                break
                
        for m in cur_mols:
            mass = get_mol_exact_mass(m)
            if abs(mass - target_mass) <= tol:
                try:
                    smi = Chem.MolToSmiles(m)
                    ik14 = MolToInchiKey(m)[:14]
                    if ik14 not in seen_ik14:
                        seen_ik14.add(ik14)
                        out_mutants.append({
                            'smiles': smi,
                            'inchikey14': ik14,
                            'mass': mass,
                            'delta_ppm': (mass - target_mass) / target_mass * 1e6,
                            'steps': len(combo),
                            'combo': combo,
                            'seed_smiles': seed_smiles,
                        })
                        if len(out_mutants) >= max_mutants:
                            return out_mutants
                except Exception:
                    continue
        if len(out_mutants) >= max_mutants or budget <= 0:
            break
            
    return out_mutants

def generate_cohort_mutants(cohort_path: str, max_mutants_per_mol: int = 100, max_mols: Optional[int] = None) -> Dict:
    import pickle
    with open(cohort_path, 'rb') as f:
        data = pickle.load(f)
    
    records = data['cohort']
    # Use top25_misses as target population
    target_ids = data['top25_misses']
    if max_mols is not None:
        target_ids = target_ids[:max_mols]
        
    print(f"Initializing KnapsackSolver (depth <= 3)...")
    solver = KnapsackSolver(max_depth=3)
    print(f"Loaded {len(solver.combos)} biogenetic transform combinations.")
    
    results = {}
    total_generated = 0
    mols_with_mutants = 0
    reach_hits = 0
    
    for idx, mid in enumerate(target_ids):
        rec = records[mid]
        target_mass = rec['target_mass']
        target_ik14 = rec['inchikey14']
        seeds = rec['fused_top'][:5]  # Top 5 retrieved analogs
        
        mol_mutants = []
        seen_ik14 = set()
        
        for seed_smi in seeds:
            sub_mutants = mutate_seed(seed_smi, target_mass, solver, max_mutants=max_mutants_per_mol - len(mol_mutants))
            for mut in sub_mutants:
                if mut['inchikey14'] not in seen_ik14:
                    seen_ik14.add(mut['inchikey14'])
                    mol_mutants.append(mut)
            if len(mol_mutants) >= max_mutants_per_mol:
                break
                
        results[mid] = {
            'molecule_id': mid,
            'target_ik14': target_ik14,
            'target_mass': target_mass,
            'target_smiles': rec['smiles'],
            'mutants': mol_mutants,
            'num_mutants': len(mol_mutants),
            'target_reached': target_ik14 in seen_ik14,
        }
        
        if mol_mutants:
            mols_with_mutants += 1
            total_generated += len(mol_mutants)
        if target_ik14 in seen_ik14:
            reach_hits += 1
            
        if (idx + 1) % 10 == 0 or (idx + 1) == len(target_ids):
            print(f"[{idx+1}/{len(target_ids)}] Evaluated {mid} | Mutants: {len(mol_mutants)} | Reach: {reach_hits}/{idx+1} ({reach_hits/(idx+1)*100:.1f}%)")
            
    print("\n" + "=" * 50)
    print(f"Total queries: {len(target_ids)}")
    print(f"Queries with >=1 mutant: {mols_with_mutants}/{len(target_ids)} ({mols_with_mutants/len(target_ids)*100:.1f}%)")
    print(f"Average mutants per query: {total_generated/len(target_ids):.1f}")
    print(f"Ground Truth Reach: {reach_hits}/{len(target_ids)} ({reach_hits/len(target_ids)*100:.1f}%)")
    print("=" * 50)
    
    return results
