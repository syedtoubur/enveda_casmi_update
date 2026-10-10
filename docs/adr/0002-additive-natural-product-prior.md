# 2. Additive Natural Product Prior and Candidate Pool Expansion

## Context
Competition test queries originate exclusively from biological extracts, but the 106M PubChem retrieval database (`casmi26-pubchem-tier`) is 98.7% synthetic lab-generated decoys. Filtering or replacing PubChem with the 152k LOTUS/NPAtlas table (`casmi26-natural-product-knowledge-table`) would amputate candidate recall on over 550,000 legitimate natural products (lipids, microbial metabolites, and non-LOTUS secondary metabolites).

## Decision
We retain the complete 106M PubChem search space to preserve 100% candidate recall, but implement an additive Natural Product Prior ($\gamma \cdot \mathbb{I}(c \in \text{LOTUS/NPAtlas})$) that heavily boosts verified natural products over synthetic lab decoys during candidate ranking. Additionally, we perform Candidate Pool Expansion by injecting the 2,715 natural products present in LOTUS/NPAtlas that were absent from the 710k public candidate pool.

## Consequences
- Protects Class 2 retrieval recall on diverse metabolites and lipids.
- Naturally suppresses synthetic combinatorial decoys inside candidate precursor mass windows without hard cutoffs.
- Expands the absolute theoretical retrieval ceiling by +2,715 known natural product connectivities.
