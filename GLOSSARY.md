# CASMI 2026

Enveda CASMI 2026: 2D molecular structure identification from LC-MS/MS mass spectra evaluated on MRR@25 of tautomer-canonical InChIKey14.

## Language

**Class 1 Molecule**:
A query molecule whose MS/MS spectrum matches a known reference library spectrum within a tight ppm window.
_Avoid_: Library hit, known spectrum

**Class 2 Molecule**:
A query molecule that exists in chemical databases (PubChem, COCONUT, ChEBI) but lacks reference spectra in the training library.
_Avoid_: Database molecule, catalog compound

**Class 3 Molecule**:
A query molecule whose 2D molecular graph is completely absent from PubChem and public reference databases, giving standard database retrieval a 0.0 MRR ceiling.
_Avoid_: Novel compound, out-of-distribution molecule

**Rank Cannibalization**:
The degradation of overall MRR caused when speculative generated candidates displace correct database retrieval candidates from high ranks (specifically Rank 1).
_Avoid_: Friendly fire, decoy inflation

**Anchored Mutator**:
A generative pipeline that harvests mass-shifted spectral analogs from the candidate pool and applies signed biochemical mass transformations (+/- O, +/- CH3, +/- sugars) to reach the query precursor mass.
_Avoid_: Graph generator, random mutator

**Formula Isolation Tag**:
A suffix (`|gen`) appended to generated molecular formulas that prevents forward MS simulators from clustering and swapping generated regioisomers with retrieved same-formula database candidates.
_Avoid_: Synthetic tag, dummy formula

**Natural Product Prior**:
An additive scoring bonus applied to candidate structures identified in natural product repositories (LOTUS, NPAtlas, COCONUT) that elevates them over synthetic compounds.
_Avoid_: Popularity filter, NP mask

**Candidate Pool Expansion**:
The process of injecting curated, missing chemical structures into the local candidate pool without restricting the underlying chemical search space.
_Avoid_: Database replacement, catalog pruning

**Lab-Generated Decoy**:
A synthetic combinatorial or medicinal chemistry molecule present in massive databases like PubChem that shares the precursor mass of a natural extract query but is biologically irrelevant.
_Avoid_: False positive, synthetic artifact

