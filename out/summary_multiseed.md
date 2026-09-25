# Multi-seed replication (mean +/- std over 10 seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

| Method | mean dev (m) | adaptation (s) | #evade | min dist (m) | energy | success (%) | OPE |
|---|---|---|---|---|---|---|---|
| Pure Pursuit | 19.19+/-1.37 | 11.9+/-5.2 | 9.8+/-1.4 | 2.61+/-0.95 | 0.157+/-0.005 | 40.72+/-3.09 | 0.325+/-0.120 |
| Line of Sight | 14.90+/-1.58 | 12.4+/-1.5 | 12.3+/-1.7 | 2.35+/-0.69 | 0.155+/-0.007 | 48.11+/-3.18 | 0.445+/-0.084 |
| Vector Field | 22.12+/-3.27 | 26.1+/-7.6 | 6.5+/-1.3 | 8.64+/-3.36 | 0.148+/-0.005 | 38.83+/-8.16 | 0.561+/-0.227 |
| Nonlinear Stabilization | 21.23+/-2.48 | 20.4+/-3.6 | 7.5+/-0.9 | 6.25+/-1.89 | 0.149+/-0.005 | 39.35+/-5.69 | 0.497+/-0.188 |
| Proposed AI | 23.77+/-1.21 | 23.2+/-3.9 | 6.8+/-1.1 | 8.42+/-4.54 | 0.139+/-0.003 | 35.01+/-2.29 | 0.550+/-0.116 |

## Exact paired permutation tests (Proposed AI vs baseline, per-seed paired OPE, H0: equal means)

| Baseline | dOPE (AI - base) | p (exact) | verdict (a=0.05) |
|---|---|---|---|
| Pure Pursuit | +0.225 | 0.0098 | significant |
| Line of Sight | +0.105 | 0.0020 | significant |
| Vector Field | -0.011 | 0.9141 | not significant |
| Nonlinear Stabilization | +0.053 | 0.5391 | not significant |

## Same test applied to every metric (AI - baseline, 10-seed paired means)

| Metric | Pure Pursuit | Line of Sight | Vector Field | Nonlinear Stabilization |
|---|---|---|---|---|
| mean_dev | +4.577 (p=0.0020) | +8.862 (p=0.0020) | +1.646 (p=0.2188) | +2.535 (p=0.0039) |
| adaptation | +11.283 (p=0.0059) | +10.717 (p=0.0020) | -2.917 (p=0.2031) | +2.767 (p=0.0117) |
| n_evade | -3.000 (p=0.0020) | -5.567 (p=0.0020) | +0.233 (p=0.7930) | -0.767 (p=0.1250) |
| min_dist | +5.809 (p=0.0020) | +6.079 (p=0.0020) | -0.219 (p=0.8809) | +2.177 (p=0.2031) |
| mean_energy | -0.018 (p=0.0020) | -0.016 (p=0.0020) | -0.009 (p=0.0039) | -0.010 (p=0.0059) |
| success | -5.714 (p=0.0020) | -13.101 (p=0.0020) | -3.821 (p=0.2109) | -4.341 (p=0.0059) |

Proposed AI OPE rank per seed: #1: 5/10, #2: 3/10, #3: 1/10, #4: 1/10