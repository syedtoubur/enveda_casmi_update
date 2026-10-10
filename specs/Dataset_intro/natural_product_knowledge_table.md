CASMI26 Natural Product Knowledge Table
One row per known natural product: 152,717 connectivities from LOTUS and NPAtlas, each with its chemical class, the organisms that make it, the key the CASMI 2026 scorer would compute, and how well PubChem documents it.

The hidden test of the competition comes from biological extracts. Only 8.2% of these natural products have a training spectrum (12,547 of 152,717), while 98% are present in the public candidate pool, so most of them must be identified without a reference spectrum. This table describes that chemistry: what classes it contains, which organisms produce it, and how much is known about each compound.

Quick start
import pandas as pd
D = '/kaggle/input/casmi26-natural-product-knowledge-table'
np_table = pd.read_parquet(f'{D}/natural_products.parquet')      # sorted by neutral monoisotopic mass

# candidates within 10 ppm of a neutral mass, with their chemical class and documentation
m = 354.1467
hits = np_table[(np_table.mono_mass - m).abs() < m * 10e-6]
print(hits[['smiles', 'np_superclass', 'kingdoms', 'n_organisms', 'pubmed_articles', 'in_casmi_train']])

# which plant genera make a given structure
genus = pd.read_parquet(f'{D}/lotus_structure_genus.parquet')
print(genus[genus.ik14 == hits.ik14.iloc[0]].sort_values('n_references', ascending=False).head())
Files
File	Rows	Content
natural_products.parquet	152,717	One row per InChIKey connectivity block; columns described below.
lotus_structure_genus.parquet	291,660	LOTUS structure–genus pairs with kingdom, family, species count and reference count.
SUMMARY.json		Row counts, overlaps and the RDKit version.
build_np.py		The complete build script.
Columns of natural_products.parquet
Column	Meaning
ik14	InChIKey connectivity block (first 14 characters) from the source database.
scorer_key	Key computed the way the competition scorer does it: InChIKey14 of the RDKit 2026.03.3 canonical tautomer.
smiles, formula, mono_mass, n_heavy	Stereo-free structure, formula, neutral monoisotopic mass (Da), heavy atoms.
in_lotus, in_npatlas	Source membership (28,763 compounds are in both).
in_casmi_pool, in_casmi_train	Present in the public 710,701-structure candidate pool / has a training spectrum.
np_pathway, np_superclass, np_class	NPClassifier biosynthetic pathway (7), superclass (76) and class (664), from LOTUS.
n_organisms, n_references, n_stereoisomers	Distinct organisms, literature references and stereoisomers recorded in LOTUS.
kingdoms, top_families, top_genera	Most frequent producing taxa (up to three, separated by semicolons).
npatlas_id, npatlas_name, npatlas_origin, npatlas_genera, npatlas_first_year	NPAtlas record: microbial origin (bacterium or fungus), genus and year of first report.
pubchem_cid, pubchem_substances, pubmed_articles, patents	Most-deposited PubChem CID of the connectivity and counts summed over its CIDs (PubChem release 2026-09-27).
What the table shows
Classes. Terpenoids 58,356 · shikimates and phenylpropanoids 27,638 · alkaloids 19,574 · polyketides 19,371 · amino acids and peptides 7,809 · fatty acids 7,410 · carbohydrates 1,458.
Organisms. Plants dominate LOTUS (about 94,000 structures reported only from Archaeplastida); NPAtlas adds bacterial and fungal metabolites. Two thirds of the LOTUS structures are reported from a single organism.
Documentation. 39% of the compounds have at least one PubMed article linked in PubChem and 35% appear in a patent, far above a typical PubChem structure (3.5% with a PubMed article).
Coverage. 146,661 structures fall in the 150–1,250 Da range of the public PubChem tier; 2,715 are absent from the public candidate pool.
Ideas for use
Class-aware validation: report MRR@25 by np_pathway to see which chemistry a model handles.
Priors: producing taxa, class frequencies and documentation counts as ranking features for same-formula isomers.
Candidate expansion: the 2,715 structures missing from the public pool.
Analogue reasoning: group candidates by np_superclass to find close relatives of a library hit.
How it was built
build_np.py merges the LOTUS frozen metadata of 2023-01-06 (Zenodo 7534071) and the NPAtlas download retrieved on 2026-09-30 by InChIKey connectivity block, recomputes structures, formulas, masses and scorer keys with RDKit 2026.03.3, and joins PubChem counts from the Compound Extras release of 2026-09-27 (CID-SID, CID-PMID, CID-Patent, CID-InChI-Key). Pool and training membership come from the public candidate pool and identity folds.

Terms and credit
LOTUS (Rutz et al., eLife 2022) is released under CC0. NPAtlas (van Santen et al., Nucleic Acids Research 2022) is released under CC BY 4.0; please cite it when you use the npatlas_* columns. PubChem data are distributed by NCBI for free reuse. This compilation is shared under CC BY 4.0. No competition spectra or test data are included.


View less
Usability
10.00

License
Attribution 4.0 International (CC BY 4.0)

Expected update frequency
Never

Tags
Biology
Science and Technology
Chemistry
Biotechnology
README.md(5.65 kB)
About this file

Suggest Edits
Dataset card.

CASMI26 Natural Product Knowledge Table
One row per known natural product: 152,717 connectivities from LOTUS and NPAtlas, each with its chemical class, the organisms that make it, the key the CASMI 2026 scorer would compute, and how well PubChem documents it.

The hidden test of the competition comes from biological extracts. Only 8.2% of these natural products have a training spectrum (12,547 of 152,717), while 98% are present in the public candidate pool, so most of them must be identified without a reference spectrum. This table describes that chemistry: what classes it contains, which organisms produce it, and how much is known about each compound.

Quick start
import pandas as pd
D = '/kaggle/input/casmi26-natural-product-knowledge-table'
np_table = pd.read_parquet(f'{D}/natural_products.parquet')      # sorted by neutral monoisotopic mass

# candidates within 10 ppm of a neutral mass, with their chemical class and documentation
m = 354.1467
hits = np_table[(np_table.mono_mass - m).abs() < m * 10e-6]
print(hits[['smiles', 'np_superclass', 'kingdoms', 'n_organisms', 'pubmed_articles', 'in_casmi_train']])

# which plant genera make a given structure
genus = pd.read_parquet(f'{D}/lotus_structure_genus.parquet')
print(genus[genus.ik14 == hits.ik14.iloc[0]].sort_values('n_references', ascending=False).head())
Files
File	Rows	Content
natural_products.parquet	152,717	One row per InChIKey connectivity block; columns described below.
lotus_structure_genus.parquet	291,660	LOTUS structure–genus pairs with kingdom, family, species count and reference count.
SUMMARY.json		Row counts, overlaps and the RDKit version.
build_np.py		The complete build script.
Columns of natural_products.parquet
Column	Meaning
ik14	InChIKey connectivity block (first 14 characters) from the source database.
scorer_key	Key computed the way the competition scorer does it: InChIKey14 of the RDKit 2026.03.3 canonical tautomer.
smiles, formula, mono_mass, n_heavy	Stereo-free structure, formula, neutral monoisotopic mass (Da), heavy atoms.
in_lotus, in_npatlas	Source membership (28,763 compounds are in both).
in_casmi_pool, in_casmi_train	Present in the public 710,701-structure candidate pool / has a training spectrum.
np_pathway, np_superclass, np_class	NPClassifier biosynthetic pathway (7), superclass (76) and class (664), from LOTUS.
n_organisms, n_references, n_stereoisomers	Distinct organisms, literature references and stereoisomers recorded in LOTUS.
kingdoms, top_families, top_genera	Most frequent producing taxa (up to three, separated by semicolons).
npatlas_id, npatlas_name, npatlas_origin, npatlas_genera, npatlas_first_year	NPAtlas record: microbial origin (bacterium or fungus), genus and year of first report.
pubchem_cid, pubchem_substances, pubmed_articles, patents	Most-deposited PubChem CID of the connectivity and counts summed over its CIDs (PubChem release 2026-09-27).
What the table shows
Classes. Terpenoids 58,356 · shikimates and phenylpropanoids 27,638 · alkaloids 19,574 · polyketides 19,371 · amino acids and peptides 7,809 · fatty acids 7,410 · carbohydrates 1,458.
Organisms. Plants dominate LOTUS (about 94,000 structures reported only from Archaeplastida); NPAtlas adds bacterial and fungal metabolites. Two thirds of the LOTUS structures are reported from a single organism.
Documentation. 39% of the compounds have at least one PubMed article linked in PubChem and 35% appear in a patent, far above a typical PubChem structure (3.5% with a PubMed article).
Coverage. 146,661 structures fall in the 150–1,250 Da range of the public PubChem tier; 2,715 are absent from the public candidate pool.
Ideas for use
Class-aware validation: report MRR@25 by np_pathway to see which chemistry a model handles.
Priors: producing taxa, class frequencies and documentation counts as ranking features for same-formula isomers.
Candidate expansion: the 2,715 structures missing from the public pool.
Analogue reasoning: group candidates by np_superclass to find close relatives of a library hit.
How it was built
build_np.py merges the LOTUS frozen metadata of 2023-01-06 (Zenodo 7534071) and the NPAtlas download retrieved on 2026-09-30 by InChIKey connectivity block, recomputes structures, formulas, masses and scorer keys with RDKit 2026.03.3, and joins PubChem counts from the Compound Extras release of 2026-09-27 (CID-SID, CID-PMID, CID-Patent, CID-InChI-Key). Pool and training membership come from the public candidate pool and identity folds.

Terms and credit
LOTUS (Rutz et al., eLife 2022) is released under CC0. NPAtlas (van Santen et al., Nucleic Acids Research 2022) is released under CC BY 4.0; please cite it when you use the npatlas_* columns. PubChem data are distributed by NCBI for free reuse. This compilation is shared under CC BY 4.0. No competition spectra or test data are included.