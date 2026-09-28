#!/bin/sh
# Everything that can be checked without Colab/Kaggle. Run before shipping any change.
set -e
cd "$(dirname "$0")/.."
python build/selftest.py            | tail -2
python build/test_process_query.py  | tail -2
python build/test_cohort.py casmi_rank_matrix.ipynb        | tail -1
python build/test_cohort.py casmi_rank_matrix_colab.ipynb  | tail -1
python build/test_gate.py           | tail -1
python build/test_stage_d_colab.py | tail -1
python build/test_simple_sim_gate.py | tail -1
python build/test_simple_sim_grid.py | tail -1
python build/test_simple_sim.py      | tail -1
python build/test_novel_p0.py      | tail -1
