"""Manoeuvring-energy metrics, reproduced verbatim from Section 1.4.

The paper defines instantaneous heading and a conditional "manoeuvring effort":

    psi_t     = atan2( y_t - y_{t-1}, x_t - x_{t-1} )
    E_t       = | wrap( psi_t - psi_{t-1} ) |,   wrap -> (-pi, pi]
    E_mean    = (1/T) * sum_t E_t
    E_max     = max_t E_t
    E_total   = sum_t E_t

Energy here is a *surrogate* for real battery drain (a function of how sharply
and how often the UAV turns), exactly as the paper states: "energy consumption
is evaluated ... as a conditional indicator of 'manoeuvring effort'".
"""
from __future__ import annotations

import numpy as np


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap an angle difference to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def headings(pts: np.ndarray) -> np.ndarray:
    """Heading sequence psi_t from positions (T,2).`pts[0]` yields no heading."""
    d = np.diff(pts, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def step_energy(pts: np.ndarray) -> np.ndarray:
    """Per-step |wrap(dpsi)| (T-1 values for a (T,2) trajectory)."""
    psi = headings(pts)
    return np.abs(wrap_pi(np.diff(psi)))


def energy_metrics(pts: np.ndarray) -> dict:
    """Return dict with E_total, E_mean, E_max, E_sigma (std). Matches Table 1."""
    E = step_energy(pts)
    if E.size == 0:
        return {"E_total": 0.0, "E_mean": 0.0, "E_max": 0.0, "E_sigma": 0.0, "n_steps": 0}
    return {
        "E_total": float(np.sum(E)),
        "E_mean": float(np.mean(E)),
        "E_max": float(np.max(E)),
        "E_sigma": float(np.std(E)),
        "n_steps": int(E.size),
    }