# Descriptor and identity compatibility

This atlas contains reference metadata and numerical mass-window counts. It supplies no fingerprint conversion or calibrated spectrum-to-fingerprint prediction.

| Related source | Reported descriptor contract | What remains unqualified here |
|---|---|---|
| MassBank source card | Morgan radius 2, 1,024 bits; MACCS, hex-packed | Exact producer RDKit version, chirality, hex packing/bit order and MACCS padding convention |
| Aryan's candidate/model resources | Morgan-2048 plus MACCS as described by the publisher | Matching model-output coordinates, producer parameters, serialization and row alignment |
| Legacy prvsiyan selected FPNet bank | A 10,407-coordinate concatenation: Morgan-r2/4096, Morgan-r3/4096, RDKit path/2048 and MACCS/167, then selection to 6,930 coordinates | Selection provenance and every required producer/model/candidate alignment check |

Different Morgan widths or selected coordinates do not establish a numerical conversion. A shared family name does not establish aligned bytes. In particular, avoid resizing a learned model output to a candidate width or silently treating byte-padding positions as real descriptor bits.

Use the candidate interoperability kit to exercise numerical storage contracts, then qualify the real producer independently. Shape/padding checks do not establish chemical descriptor semantics or a model's calibrated probabilities. This build reads no fingerprint column or checkpoint, and no probabilistic calibration is measured.

The join key here is the full supplied InChIKey. Its syntax check does not establish chemical validity or equivalence to a competition's scoring normalization. Preserve source keys, row positions, descriptor representations and any independently qualified scoring identity as separate evidence.

Mass windows are also a contract. This atlas uses `q` in the tolerance denominator: `q ± q*ppm/1e6`, with closed bounds. It does not use precursor m/z or a candidate-denominator formula. The supplied reference mass's neutral basis remains unverified; the related ion atlas defines theoretical ion channels but does not assign an observed ion to a reference record.
