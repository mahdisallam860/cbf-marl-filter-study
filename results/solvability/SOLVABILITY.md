# Layout solvability audit (geometry only) beside the filter-off gate

Planner: grid search (cell 0.02 m) on certainly-free cells, shortcut, every segment checked exactly; obstacles inflated by R + R_o + 0.002 m, robots kept 0.2 + 0.002 m apart; unicycle follows the polyline by turning in place (|w| <= 1.0 rad/s) and driving straight (v <= 0.5 m/s) at dt = 0.1 s; horizon 400 steps (40 s). Formation tolerance 0.1 m mean slot error (the protocol checks it only at the goal; (c) imposes it during the whole transit). BLOCKED = proven by disconnection of an over-approximated free space (any horizon); UNDETERMINED = neither a verified plan nor a proof. No arena walls (A6). Audit box (-2.6, 2.6, -2.0, 2.0), outer ring free.

| split | K | layouts (seed) | (a) each robot found / blocked / undet. | (b) team found / blocked / undet. | (c) team within tol found / blocked / undet. | (b) steps median / max | gate GOAL, selected ckpt | gate GOAL, best ckpt |
|---|---|---|---|---|---|---|---|---|
| validation | 2 | 200 (2000002) | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 152 / 180 | 32.5% | 32.5% (seed 0 it 300) |
| validation | 5 | 200 (2000005) | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 179 / 210 | 0.5% | 0.5% (seed 0 it 200) |
| validation | 8 | 200 (2000008) | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 200 (100.0%, CI 98.1-100.0) / 0 / 0 | 193 / 216 | 0.0% | 0.0% (all 100 checkpoints) |
| pilot | 2 | 500 (3000002) | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 151 / 189 | - | - |
| pilot | 5 | 500 (3000005) | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 181 / 206 | - | - |
| pilot | 8 | 500 (3000008) | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 500 (100.0%, CI 99.2-100.0) / 0 / 0 | 192 / 216 | - | - |
