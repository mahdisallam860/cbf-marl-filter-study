# Phase 1: solvability of the validation layouts at K = 1, 3, 4

Same planner, settings and status rules as `results/solvability/` (geometry only; FOUND = verified collision-free plan within 400 steps; BLOCKED = proven disconnected; UNDETERMINED = neither). (a) each robot alone, (b) the team, (c) the team within the formation tolerance during transit. K = 4 is an intermediate diagnostic density, not a study density.

| K | role | layouts (seed) | (a) found / blocked / undet. | (b) found / blocked / undet. | (c) found / blocked / undet. | (b) steps median / max | (c) found, Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| 1 | candidate study density | 200 (2000001) | 200 / 0 / 0 | 200 / 0 / 0 | 200 / 0 / 0 | 143 / 170 | 98.1-100.0% |
| 3 | candidate study density | 200 (2000003) | 200 / 0 / 0 | 200 / 0 / 0 | 200 / 0 / 0 | 162 / 201 | 98.1-100.0% |
| 4 | intermediate diagnostic | 200 (2000004) | 200 / 0 / 0 | 200 / 0 / 0 | 200 / 0 / 0 | 176 / 204 | 98.1-100.0% |
