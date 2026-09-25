# Comparative study reconstruction (Section 2)

Independent, seeded reconstruction. Per-method values are aggregated (mean over straight / circle / polyline). OPE is the empirical batch-normalised productivity coefficient (higher = better).

| Method | mean dev (m) | adaptation (s) | #evade | min dist (m) | energy (unit) | success (%) | OPE |
|---|---|---|---|---|---|---|---|
| Pure Pursuit | 18.78 | 8.5 | 10 | 2.66 | 0.159 | 39.53 | 0.228 |
| Line of Sight | 13.54 | 14.8 | 10 | 2.54 | 0.160 | 52.27 | 0.400 |
| Vector Field | 23.68 | 38.7 | 7 | 11.34 | 0.152 | 30.90 | 0.403 |
| Nonlinear Stabilization | 21.63 | 22.5 | 7 | 5.94 | 0.152 | 36.66 | 0.372 |
| Proposed AI | 24.83 | 25.7 | 5 | 10.25 | 0.134 | 36.32 | 0.626 |

**Ranking by OPE:** Proposed AI, Vector Field, Line of Sight, Nonlinear Stabilization, Pure Pursuit
