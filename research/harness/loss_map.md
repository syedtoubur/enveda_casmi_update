# Loss map of the 0.433 base (harness replay, unchanged config)

Where the base loses MRR@25 on each slice. Levels read high (L07, gate_report.md); use the buckets to aim ideas.

## C2 (n = 400, MRR@25 0.631)

| bucket | molecules | share | MRR headroom if fixed to rank 1 |
|---|---|---|---|
| rank 1 | 166 | 41.5% | +0.000 |
| rank 2-5 | 204 | 51.0% | +0.298 |
| rank 6-25 | 16 | 4.0% | +0.035 |
| not in top 25, but in some channel list | 2 | 0.5% | +0.005 |
| in no channel list (pool / retrieval miss) | 12 | 3.0% | +0.030 |

| class | n | MRR |
|---|---|---|
| 2 | 400 | 0.631 |

## C1_npx (n = 245, MRR@25 0.933)

| bucket | molecules | share | MRR headroom if fixed to rank 1 |
|---|---|---|---|
| rank 1 | 220 | 89.8% | +0.000 |
| rank 2-5 | 20 | 8.2% | +0.048 |
| rank 6-25 | 4 | 1.6% | +0.015 |
| not in top 25, but in some channel list | 1 | 0.4% | +0.004 |
| in no channel list (pool / retrieval miss) | 0 | 0.0% | +0.000 |

| class | n | MRR |
|---|---|---|
| 1 | 245 | 0.933 |

## C2_hard (n = 200, MRR@25 0.638)

| bucket | molecules | share | MRR headroom if fixed to rank 1 |
|---|---|---|---|
| rank 1 | 89 | 44.5% | +0.000 |
| rank 2-5 | 92 | 46.0% | +0.275 |
| rank 6-25 | 13 | 6.5% | +0.057 |
| not in top 25, but in some channel list | 0 | 0.0% | +0.000 |
| in no channel list (pool / retrieval miss) | 6 | 3.0% | +0.030 |

| class | n | MRR |
|---|---|---|
| 2 | 200 | 0.638 |

## C1_xlib (n = 300, MRR@25 0.929)

| bucket | molecules | share | MRR headroom if fixed to rank 1 |
|---|---|---|---|
| rank 1 | 268 | 89.3% | +0.000 |
| rank 2-5 | 22 | 7.3% | +0.038 |
| rank 6-25 | 2 | 0.7% | +0.006 |
| not in top 25, but in some channel list | 1 | 0.3% | +0.003 |
| in no channel list (pool / retrieval miss) | 7 | 2.3% | +0.023 |

| class | n | MRR |
|---|---|---|
| 1 | 300 | 0.929 |

## C2_np (n = 245, MRR@25 0.596)

| bucket | molecules | share | MRR headroom if fixed to rank 1 |
|---|---|---|---|
| rank 1 | 79 | 32.2% | +0.000 |
| rank 2-5 | 155 | 63.3% | +0.364 |
| rank 6-25 | 10 | 4.1% | +0.035 |
| not in top 25, but in some channel list | 0 | 0.0% | +0.000 |
| in no channel list (pool / retrieval miss) | 1 | 0.4% | +0.004 |

| class | n | MRR |
|---|---|---|
| 1 | 245 | 0.596 |

