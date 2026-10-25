# Shape of the verified routes (validation layouts, the gate's layouts)

Near-shortest verified routes from the solvability planner. 'around' = the rigid triangle's centroid reaches |y| >= 1.5 m (whole formation outside the obstacle band) / an individual robot leaves the band (|y| >= 1.2 m). Length ratio = route length / straight start-goal distance.

| K | layouts | (c) formation routes around the field | (c) length ratio median [p90] | (a) some robot goes around | (a) worst robot length ratio median [p90] |
|---|---|---|---|---|---|
| 2 | 200 | 0/200 | 1.05 [1.20] | 0/200 | 1.01 [1.04] |
| 5 | 200 | 2/200 | 1.33 [1.44] | 1/200 | 1.05 [1.15] |
| 8 | 200 | 10/200 | 1.46 [1.53] | 19/200 | 1.16 [1.28] |
