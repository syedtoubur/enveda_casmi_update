# G4 fix brief: 20261017-gen-rank1-keep

- kernel: `syedtoubur/casmi26-gen-rank1-keep` (round 1 of 2; log in `kaggle_output/`)
- kind: replay; line: c3; worktree: `E:/casmi/lines/c3`

## Re-push command (no --wait; add `--source <owner/slug>` for a catalog dataset)
```
python research/bin/kaggle_push.py research/c3/20261017-gen-rank1-keep "E:\casmi\lines\c3\casmi26-v22-gen1-keep.ipynb" --preset sota_0433_inference_preset --slug casmi26-gen-rank1-keep --counter "GEN1_KEEP gen1_checked" --counter "?GEN1_KEEP gen1_kept" --counter "?GEN1_KEEP gen1_moved" --source syedtoubur/casmi26-e180-keys
```

## The one change (spec.md, up to the 3rd section)
# Spec: gen-rank1-keep (R1, knob `GEN1_KEEP`)

Campaign `20261017-c3-iso`. Gate block: gate_report `### 2026-10-17: c3-iso campaign, PRE-REGISTERED`, bullet R1, plus
the rank-only gate (2026-10-13). Target cohort pseudo_c3.

### Hypothesis
Engine 1 (`generate=True`) sometimes puts a generated structure at BASE #1 (formula tagged `'|gen'`, L17). Under E006
this was 138/600 pseudo_c3 (55 correct) but only 2/245, 1/300, 6/400, 4/200 on C1_npx/C1_xlib/C2/C2_hard (K0
observation). Post-scoring then moved 44 of the 55 correct ones to ranks 2–7. Keeping BASE #1 at rank 1 when ICEBERG
also prefers it (leak-free guard; f·z is leaky on pseudo_c3) should recover most of them, with almost no C1/C2 fires.

### Knob (`E:/casmi/lines/<idea worktree>/build/casmi_post.py`, default off = byte-identical)
`GEN1_KEEP=False` in CFG. When True, per active molecule m:
- g = BASE[m] #1 as stored in the dump **before** the C8 rerank (`state['base'][m]`: smiles[0], formula[0]).
- Fire iff formula[0] ends with `'|gen'` AND g's ICE cosine (`state['ice'][m]`, same key convention as `ice_rerank`)
  is not None and is ≥ the ICE cosine of every other candidate among BASE top 5 that has one (ties fire) AND at least
  one other top-5 candidate has an ICE score (no comparison → no fire).
- Effect: the final row becomes [g] + [x for x in row if identity(x) != identity(g)], cut at 25 (metric_identity dedupe).
  Applied as the very last step of `run()` (after E2R2 and lib-isomer). Counter `stats['gen1_kept']` (fires) and
  `stats['gen1_moved']` (fires where g was not already rank 1).
- No FPNet value is read by the trigger.


## Fail reasons
- draft identical to control on 400/400 molecules (changed 0 < 1): the change never fired

## Log tail (last 80 lines)
```
casmi from /kaggle/input/datasets/ahmedberatozer/casmi26-v4b-models/code/casmi/__init__.py
cache built 817s
[Library] 2,539,608 spectra, 275,810 structures, 133,228,108 peaks (1.5s)
[Pool] 710,701 structures, nbits 10226, frag yes (8.7s)
engine FP bank: /kaggle/input/datasets/ahmedberatozer/casmi26-fpnet-full1/fpnet_full1.pt
[Engine] reps 445,686 (0.5s)
derivation prior self-test mismatches: 0
engine ready; ranker features 160 boosters 4 846s
v19 pool popularity re-rank, mu 0.15 pool 710701
  12/12 molecules 897s
errors 0 | popularity re-ranked lists 12
ICE_UNION: engine candidates added to ICE input: 772
ICE input molecules 12 candidates 772 897s
[claw] site preseed rc 0 
[claw] site ready; substituted ['rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl.ice'] -> ['rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl.ice']
[ice     0.2s] stubbed (absent) modules: ['pygmtools']
[ice    20.2s] ms_pred tautomer mode v2 cache /tmp/ice_ms_pred_cache
[ice    20.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True)
[ice    65.7s] chunk: 11 mols 1524 jobs 38.4s (11/11 covered mols scored)
[ice    65.7s] status ok {'n_mols': 12, 'n_mols_covered': 11, 'n_mols_scored': 11, 'n_cands': 772, 'n_scored': 754}

{'status': 'ok', 'n_mols': 12, 'n_mols_covered': 11, 'n_mols_scored': 11, 'n_cands': 772, 'n_scored': 754, 'seconds': 65.72, 'pred_per_s': 40.12, 'device': 'cuda', 'rdkit': '2025.03.6', 'max_mem_gb': 3.75}
ICE meta {"status": "ok", "n_mols": 12, "n_mols_scored": 11, "n_mols_covered": 11, "n_cands": 772, "n_scored": 754, "errors": [], "rdkit": "2025.03.6", "torch": "2.10.0+cu128", "device": "cuda", "t_load": 20.83, "n_pred": 1524, "n_pred_fail": 0, "t_pred": 37.99, "n_bad_smiles": 0, "pred_per_s": 40.12, "max_mem_gb": 3.75, "seconds": 65.72, "log_tail": ["[ice     0.2s] stubbed (absent) modules: ['pygmtools']", "[ice    20.2s] ms_pred tautomer mode v2 cache /tmp/ice_ms_pred_cache", "[ice    20.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True)", "[ice    65.7s] chunk: 11 mols 1524 jobs 38.4s (11/11 covered mols scored)"]}
v29 GLACIER [M+H]+ only: molecules 11 of 12
[gl     0.1s] site ready (marker matches): /kaggle/working/ice_site
[gl     0.1s] stubbed (absent) modules: ['pygmtools']
[gl    14.8s] GLACIER loaded: 15,108,154 params 
[gl    14.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True, threads 2)
/usr/local/lib/python3.12/dist-packages/torch/nn/modules/transformer.py:531: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(
[gl    37.9s] chunk: 11 mols 1524 jobs 18.1s (11/11 covered mols scored)
[gl    37.9s] status ok {'n_mols': 11, 'n_mols_covered': 11, 'n_mols_scored': 11, 'n_cands': 754, 'n_scored': 754}

{'status': 'ok', 'n_mols': 11, 'n_mols_covered': 11, 'n_mols_scored': 11, 'n_cands': 754, 'n_scored': 754, 'seconds': 37.94, 'pred_per_s': 87.45, 'device': 'cuda', 'rdkit': '2025.03.6', 'max_mem_gb': 0.32}
GL meta {"model": "glacier", "status": "ok", "n_mols": 11, "n_mols_scored": 11, "n_mols_covered": 11, "n_cands": 754, "n_scored": 754, "errors": [], "rdkit": "2025.03.6", "torch": "2.10.0+cu128", "device": "cuda", "t_load": 14.76, "n_pred": 1524, "n_pred_fail": 0, "n_feat_fail": 0, "t_pred": 17.43, "n_bad_smiles": 0, "pred_per_s": 87.45, "bs_final": 32, "max_mem_gb": 0.32, "seconds": 37.94, "log_tail": ["[gl     0.1s] site ready (marker matches): /kaggle/working/ice_site", "[gl     0.1s] stubbed (absent) modules: ['pygmtools']", "[gl    14.8s] GLACIER loaded: 15,108,154 params ", "[gl    14.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True, threads 2)", "[gl    37.9s] chunk: 11 mols 1524 jobs 18.1s (11/11 covered mols scored)"]}
ICE rerank stats {'molecules': 0, 'changed_top25': 0, 'changed_top1': 0} 1008s
GL rerank stats {'molecules': 0, 'changed_vs_ice_top25': 0, 'changed_vs_ice_top1': 0} 1008s
merge/promote self-test OK
merge stats {'untouched': 12, 'gentle': 0, 'aggressive': 0, 'no_pc': 0, 'promoted': 0}
CLAW STAGE HEALTH | ICE NOT RUN | GL NOT RUN
CLAW STAGE SCORED | ice=11 gl=11
done (400, 2) 1008s
V1 inference complete; applying separately audited final pass next.
{
  "experiment": "V1.1 metric-safe tail",
  "rdkit": "2026.03.3",
  "hidden_rerun": false,
  "active_molecules": 12,
  "output_rows": 400,
  "changed_molecules": 1,
  "invalid_removed": 0,
  "equivalent_duplicates_removed": 0,
  "existing_candidates_promoted": 0,
  "ranked_candidates_added": 2,
  "dreams_candidates_added": 0,
  "monotone_rank_assertions_passed": true,
  "metric_source_sha256": "c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201",
  "baseline_sha256": "1a85c0ecb4d3dbc6e9bcfdf2720fed604e66dd393d00480f9b41fe13192ad44e",
  "submission_sha256": "56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880",
  "runtime_seconds": 1019.7,
  "validation_note": "Visible test is a smoke check only and overlaps training; no generalization score claimed."
}
{'experiment': 'metric_identity_fusion', 'changed_molecules': 0, 'changed_top1': 0, 'reference_sha256': '56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880', 'submission_sha256': '56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880', 'measured_score': None}
NO-OP: predictions match the reference; do not spend a second submission on this file.
E004_VERIFICATION {"experiment": "E004", "source_version_id": 356740494, "source_version": 80, "source_code_sha256": "eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2", "runtime_input_path": "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra", "input_ids": ["m_005e53", "m_006153", "m_00b5aa", "m_01e922", "m_0259d4", "m_02d188", "m_0318b5", "m_043e86", "m_050bf1", "m_05b52e", "m_068354", "m_06e13d", "m_06f339", "m_070985", "m_074e83", "m_089267", "m_08d7f1", "m_0a8bf5", "m_0af51a", "m_0c4e10", "m_0cc9ea", "m_0d08be", "m_0d2197", "m_0d3550", "m_0d463f", "m_0ea8a3", "m_0ee46b", "m_0f05e7", "m_0fe9d9", "m_108fdb", "m_12945e", "m_13152b", "m_134d4a", "m_145679", "m_153108", "m_156305", "m_15dab4", "m_163bfd", "m_163da0", "m_1700bc", "m_1712db", "m_1777b7", "m_1813a1", "m_18f297", "m_1a0749", "m_1a8b29", "m_1b1573", "m_1b27e6", "m_1beae5", "m_1d7bdd", "m_1de28b", "m_1f86e6", "m_20b8f2", "m_20c386", "m_214040", "m_217b19", "m_21cd44", "m_2363c0", "m_23ff71", "m_254f59", "m_25a38e", "m_25cfa1", "m_26bf54", "m_273696", "m_27df2d", "m_293ec0", "m_2a0e17", "m_2aa8ea", "m_2ac2ed", "m_2b524a", "m_2b7513", "m_2c1f87", "m_2c2340", "m_2d1eba", "m_2e265c", "m_2efd6f", "m_2f1128", "m_2f9ab1", "m_30024e", "m_31ebaa", "m_328338", "m_347e26", "m_34c00a", "m_34c315", "m_36d3ce", "m_370436", "m_3883ef", "m_38941f", "m_3a5731", "m_3aa487", "m_3ae355", "m_3aecdc", "m_3ba5ce", "m_3ce578", "m_3d62dd", "m_3db789", "m_3dc915", "m_3f6bab", "m_3f8028", "m_3ff7be", "m_402004", "m_417c79", "m_43683b", "m_436a8d", "m_436c77", "m_43e809", "m_444a56", "m_44cb82", "m_451ad6", "m_45d613", "m_466863", "m_4690d3", "m_471696", "m_47617a", "m_478c2e", "m_47ae67", "m_47e099", "m_48bfe1", "m_4b55af", "m_4b86b5", "m_4bb17c", "m_4bc064", "m_4fb975", "m_517c87", "m_53dfe2", "m_55fde0", "m_5667e2", "m_568a04", "m_56bcbe", "m_572a36", "m_57bec0", "m_57f31b", "m_59b1bd", "m_59bced", "m_5ae66f", "m_5bfac3", "m_5c710b", "m_5c8d41", "m_5da1ef", "m_5dda21", "m_5fef1c", "m_6007c7", "m_60e7b1", "m_6105c1", "m_610a9f", "m_612431", "m_62973b", "m_62dd85", "m_637284", "m_64030c", "m_64306c", "m_65a7ff", "m_6817b1", "m_6b66aa", "m_6d224e", "m_6d3dc0", "m_6d9287", "m_6e1c94", "m_6ebcf9", "m_6f127a", "m_6f3a5c", "m_6f414f", "m_6f45e5", "m_70464b", "m_70fe91", "m_70ff2a", "m_7108be", "m_7125ae", "m_741f2c", "m_75100f", "m_75dece", "m_762dcf", "m_76b47b", "m_76ffc5", "m_773a95", "m_77b2b0", "m_7874d0", "m_78c402", "m_78cd05", "m_79cf07", "m_7b993b", "m_7d4c58", "m_7e912f", "m_7ee70f", "m_80401f", "m_8071e1", "m_808712", "m_80b537", "m_8114e3", "m_82f2e2", "m_83ac87", "m_83e861", "m_83fdb3", "m_847112", "m_875d74", "m_88505f", "m_88a263", "m_895051", "m_89a6f2", "m_89b9d2", "m_8b6b8e", "m_8bb047", "m_8c72d4", "m_8d19fa", "m_8d374f", "m_8d75c7", "m_8db7e3", "m_8e45f7", "m_8eaf01", "m_9068e1", "m_907412", "m_90baa7", "m_91512a", "m_916daf", "m_921c43", "m_930a3b", "m_935b22", "m_93b6a1", "m_94a765", "m_95e8da", "m_966da0", "m_97693a", "m_97cca9", "m_982aac", "m_987cff", "m_9910fd", "m_997dbc", "m_99e777", "m_9b41d0", "m_9b8030", "m_9b95d4", "m_9c4788", "m_9dc5c4", "m_9e99f7", "m_9f1fb6", "m_9fcc62", "m_a06450", "m_a0edff", "m_a12514", "m_a3c949", "m_a5509a", "m_a5bb0e", "m_a5d8e5", "m_a87b11", "m_a94f72", "m_a98076", "m_aa5754", "m_ab3260", "m_ab91a8", "m_abbd02", "m_acdfec", "m_ace790", "m_acfc1e", "m_aec773", "m_aef2b4", "m_b047da", "m_b084eb", "m_b0ce2a", "m_b11a90", "m_b4039c", "m_b43690", "m_b51204", "m_b53b6e", "m_b6b7fe", "m_b70863", "m_b7be8a", "m_b7d932", "m_b81b9d", "m_b8415b", "m_b8a6f4", "m_b8f328", "m_b9f2b4", "m_baccbd", "m_bb75e4", "m_bbb21f", "m_bbc0aa", "m_bcb981", "m_bcd97d", "m_bcdd81", "m_bd0de9", "m_bd907b", "m_bda18e", "m_bf639c", "m_bf6cfe", "m_c1b10a", "m_c1ec62", "m_c2176f", "m_c2313f", "m_c30e40", "m_c3e52e", "m_c50497", "m_c567df", "m_c57b9d", "m_c63e89", "m_c772b8", "m_c886af", "m_c8934f", "m_c89ffd", "m_c933b7", "m_c9a4bc", "m_c9c302", "m_c9c422", "m_cad5a6", "m_cc35ab", "m_cc95e9", "m_cd50d8", "m_cd6458", "m_ce56d4", "m_ced31c", "m_cfb3df", "m_d01f54", "m_d07b86", "m_d1e5ef", "m_d1ecad", "m_d258c4", "m_d274b6", "m_d3998e", "m_d4c45b", "m_d56a4e", "m_d6c1c0", "m_d74296", "m_d7ad1b", "m_d7ecc5", "m_d86693", "m_d90c36", "m_daf5f1", "m_db2018", "m_db2c0e", "m_dbd1b2", "m_dbd5d0", "m_dc5eb2", "m_dc7be7", "m_dd5487", "m_de93b1", "m_ded5eb", "m_e138be", "m_e17598", "m_e1bfd4", "m_e1d7ad", "m_e1f9f4", "m_e2993b", "m_e3b936", "m_e44302", "m_e4865d", "m_e5050e", "m_e58af8", "m_e5b5fa", "m_e6b2c8", "m_e6bed4", "m_e73d71", "m_e77f56", "m_e7af25", "m_e80818", "m_e8359d", "m_e8d835", "m_e993ec", "m_ea2f71", "m_ea7270", "m_eaecd7", "m_ebcb8a", "m_ec2936", "m_ec5045", "m_ec755b", "m_ecbfb5", "m_ed201b", "m_edaa12", "m_edccdd", "m_edd5cd", "m_eff2fa", "m_f0bee0", "m_f10ba9", "m_f1a045", "m_f22b14", "m_f22bed", "m_f24d72", "m_f2d901", "m_f55e93", "m_f5c42c", "m_f7002e", "m_f7302f", "m_f82743", "m_f858ee", "m_f866a0", "m_f8bf38", "m_f8d39e", "m_f8ef30", "m_fb2bcc", "m_fb6996", "m_fbf20a", "m_fd4556", "m_fd693e", "m_fe6eaa", "m_fe9649", "m_fea652", "m_feb5c1", "m_ff32bb", "m_ff4ed5", "m_ff5db9", "m_ff7740", "m_ff926d"], "python": "3.12.13", "rdkit": "2026.03.3", "gpu": "Tesla T4\nTesla T4", "is_full_run": false, "processed_ids": ["m_005e53", "m_006153", "m_00b5aa", "m_01e922", "m_0259d4", "m_02d188", "m_0318b5", "m_043e86", "m_050bf1", "m_05b52e", "m_068354", "m_06e13d"], "processed_molecules": 12, "input_molecules": 400, "output_rows": 400, "placeholder_fill_rows": 388, "base_errors": 0, "base_nonempty": 12, "engine2_nonempty": 12, "pubchem_nonempty": 12, "main_rdkit": "2026.03.3", "stages": {"ice": {"status": "ok", "n_mols": 12, "n_mols_scored": 11, "n_mols_covered": 11, "n_cands": 772, "n_scored": 754, "errors": [], "rdkit": "2025.03.6", "torch": "2.10.0+cu128", "device": "cuda", "t_load": 20.83, "n_pred": 1524, "n_pred_fail": 0, "t_pred": 37.99, "n_bad_smiles": 0, "pred_per_s": 40.12, "max_mem_gb": 3.75, "seconds": 65.72, "log_tail": ["[ice     0.2s] stubbed (absent) modules: ['pygmtools']", "[ice    20.2s] ms_pred tautomer mode v2 cache /tmp/ice_ms_pred_cache", "[ice    20.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True)", "[ice    65.7s] chunk: 11 mols 1524 jobs 38.4s (11/11 covered mols scored)"]}, "gl": {"model": "glacier", "status": "ok", "n_mols": 11, "n_mols_scored": 11, "n_mols_covered": 11, "n_cands": 754, "n_scored": 754, "errors": [], "rdkit": "2025.03.6", "torch": "2.10.0+cu128", "device": "cuda", "t_load": 14.76, "n_pred": 1524, "n_pred_fail": 0, "n_feat_fail": 0, "t_pred": 17.43, "n_bad_smiles": 0, "pred_per_s": 87.45, "bs_final": 32, "max_mem_gb": 0.32, "seconds": 37.94, "log_tail": ["[gl     0.1s] site ready (marker matches): /kaggle/working/ice_site", "[gl     0.1s] stubbed (absent) modules: ['pygmtools']", "[gl    14.8s] GLACIER loaded: 15,108,154 params ", "[gl    14.8s] model loaded on cuda (rdkit 2025.03.6, torch 2.10.0+cu128, numba True, threads 2)", "[gl    37.9s] chunk: 11 mols 1524 jobs 18.1s (11/11 covered mols scored)"]}}, "hashes": {"submission.csv": "56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880", "baseline_submission.csv": "1a85c0ecb4d3dbc6e9bcfdf2720fed604e66dd393d00480f9b41fe13192ad44e", "submission_reference.csv": "56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880", "candidate_audit.json": "b66598d2c4dbb544316b0dd3656c290b93737a96b47a2777d6eb6c78dc1f7cce", "experiment_audit.json": "f1a9989791fb8c5fac196a4cf71de3265f4e0e1388c7a5e8ad798e34b0811b08", "e004_candidate_inputs.json": "e06e2c64807b3249f2a53b454f309481607b064d84aa6987a7e61c23c559f2e4"}, "elapsed_seconds": 1033.65, "limitations": ["Visible smoke is not independent validation.", "Author 0.436 is not this reproduction score.", "Whole v80 recipe differs from E003 in multiple factors."]}
v7c FRAG gap molecules 1 of 400 0s
FACTORIAL_VERIFICATION {"experiment": "E006", "window": 40, "ice_fallback": true, "source_experiment": "E004", "base_report_file": "e004_verification.json", "control_file": "submission_e004_control.csv", "control_sha256": "56425c57b69e7e3dabdd74f7ceb90087feb916fa3249701516e4af1a3948b880", "submission_sha256": "d7a437210bd6a1dd9d152eb07f92f9a23f8e57cd652da5ebaa9b05e36066e366", "processed_ids": ["m_005e53", "m_006153", "m_00b5aa", "m_01e922", "m_0259d4", "m_02d188", "m_0318b5", "m_043e86", "m_050bf1", "m_05b52e", "m_068354", "m_06e13d"], "output_rows": 400, "changed_rows": 1, "changed_top1": 0, "fallback_rows": 0, "fallback_finite_rows": 0, "fallback_changed_rows": 0, "fallback_changed_top1": 0, "same_run_control_reconstructed": true, "provenance": {"base_notebook_sha256": "1f0f9b3026a93ddb5ad6fa6993c8ac0a4bfaddc54594eb1dd6dda7a20d8ef0d6", "base_source_code_sha256": "eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2", "helper_source_sha256": "a027a69ddda2c2e14639e3ec5196d5f1eb2ab57cba10ef929723148f31fc8f34", "source_script_version_id": 356740494, "ice_policy_source_script_version_id": 356785434, "ice_policy_source": "samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438/1"}, "limitations": ["Visible smoke is not independent validation.", "An inactive fallback in visible smoke says nothing about hidden benefit."]}
E2R2 reads /kaggle/working/submission.csv 17224 bytes
E2R2_APPLIED {"mode": "rank2", "key": "metric_identity", "is_rerun": false, "rows": 400, "eligible": 12, "inserted": 0, "moved_from": {"2": 1}, "same_as_rank1": 11, "no_eng": 0, "invalid_x2": 0, "already_rank2": 1, "skipped": 388, "changed_rows": 0, "tail_dropped": 0, "extra_twins_dropped": 0, "base_sha256": "d7a437210bd6a1dd9d152eb07f92f9a23f8e57cd652da5ebaa9b05e36066e366", "submission_sha256": "d7a437210bd6a1dd9d152eb07f92f9a23f8e57cd652da5ebaa9b05e36066e366"}
GEN1_KEEP gen1_checked 12
GEN1_KEEP gen1_kept 0
GEN1_KEEP gen1_moved 0
/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
  cells[i][c] = re.sub('\\\\\|', '|', cell)
/usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
  text = re.sub(r'_', '\_', text) # Escape underscores in display text
[NbConvertApp] Converting notebook __notebook__.ipynb to notebook
[NbConvertApp] Writing 273393 bytes to __notebook__.ipynb
[NbConvertApp] Converting notebook __notebook__.ipynb to html
[NbConvertApp] Writing 1003635 bytes to __results__.html
```

## gates.md
| Gate | When | Result | Evidence | CU |
|---|---|---|---|---|
| G1 | 2026-10-17 20:00 | PASS | compare.py V22 base vs casmi26-v22-gen1-keep.ipynb (build/patch_gen1_keep.py): 22 vs 23 code cells; only C12 (BASE top-5 snapshot before the C8 rerank loop) MODIFIED and C22 (POST_SRC + GEN1 final cell) ADDED; 21 cells identical |  |
| G2 | 2026-10-17 20:00 | PASS | `python build/test_patch_gen1_keep.py` (3 tests: only named cells differ; final cell no-op without `\|gen`; pins an ICE-agreeing gen, not when ICE disagrees) in E:/casmi/lines/c3, branch idea/20261017-gen-rank1-keep-ship 3846918. run_tests.py has no c3 choice (exploit\|explore only; loop code out of scope), so the suite was not run through it. casmi_post knob tests (374f4c3) not rerun: test_casmi_post.py has no runner |  |
| G4 | 2026-10-17 20:03 | BLOCKED | preflight not clean: [R08] harness_result.json: knobs ['E2R2', 'FRAG_GAP', 'FUSION_WINDOW', 'GEN1_KEEP', 'ICE_FALLBACK', 'METRIC_FUSION', 'REL_PROMOTE'] have no family in research/harness/backtests.yaml |  |
| G4 | 2026-10-18 00:28 | BLOCKED | sources (L30): /kaggle/input/casmi26-e180-keys/e180_only_ik14.txt -> needs syedtoubur/casmi26-e180-keys; /kaggle/input/casmi26-e180-keys/e180_only_ik14.txt\ -> needs syedtoubur/casmi26-e180-keys; casmi26-e180-keys -> needs syedtoubur/casmi26-e180-keys |  |
| G4 | 2026-10-18 00:29 | PUSHED | syedtoubur/casmi26-gen-rank1-keep |  |
| G4 | 2026-10-18 01:29 | TIMEOUT | syedtoubur/casmi26-gen-rank1-keep still queued after 3600s; kaggle_push.py --check finalizes it later |  |
| G4 | 2026-10-18 02:46 | FAIL (complete; draft identical to control on 400/400 molecules (changed 0 < 1): the change never fired) | syedtoubur/casmi26-gen-rank1-keep; counters: GEN1_KEEP gen1_checked 12 / GEN1_KEEP gen1_kept 0 / GEN1_KEEP gen1_moved 0 |  |

## Fix scope
- The fix agent may change the patch code in the line worktree (`E:/casmi/lines/<line>`), then rerun G1 and G2.
- It may add push sources **that already exist in `dataset_catalog.json`** (`kaggle_push.py --source <owner/slug>`).
- A dataset missing from the catalog, or a fix that would change the idea itself: write to `questions.md`, set STATUS `needs-human`, stop.
- For a **control** idea only sources and metadata may change, never the notebook code.
- The catalog itself is never edited.

