# 1. Direct In-Process Integration for Class-3 Pipeline

## Context
Previous iterations of the Class-3 mutator (`c3v2`, `mut_runner.py`) wrote standalone Python scripts to `/kaggle/working/mut/` and invoked an external CLI subprocess with 15 environment variables to isolate package environments. In practice, this created severe disk I/O overhead, serialization fragility (JSON round-trips), silent worker failures under Kaggle RAM pressure, and untraceable exceptions.

## Decision
We replace the external subprocess relay with a direct, in-process Python module (`c3_mutator`). The module executes inside the notebook runtime, receives structured memory objects (precursor targets, analog anchors, FPNet logits $z$), and returns scored candidate dataclasses directly without intermediate disk writes.

## Consequences
- Eliminates disk serialization bottlenecks and filesystem relay bugs.
- Provides immediate stack traces and line-number errors on failure.
- Requires ensuring that in-process execution remains within Kaggle's 16GB CPU RAM and 16GB GPU VRAM budgets.
