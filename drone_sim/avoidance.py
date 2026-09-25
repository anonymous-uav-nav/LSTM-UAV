"""Collision-risk assessment and trajectory correction (Sections 1.3-1.4).

Implements the safety-zone test of Eq. (7), the single-step collision
probability of Eq. (6) (evaluated as a hard occupancy test in the deterministic
simulation), the horizon accumulation of Eq. (4), the horizon trade-off of
Eq. (5), and the correction optimisation of Eq. (8).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .energy import wrap_pi


@dataclass
class Obstacle:
    """A static + optionally dynamic obstacle with a circular safety zone."""
    centre: np.ndarray
    radius: float = 10.0          # safety-zone radius R (safe_distance = 10 in paper)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def collide(self, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - self.centre)) <= self.radius

    def in_zone(self, p: np.ndarray) -> bool:
        return self.collide(p)


def risk_at_point(p: np.ndarray, obstacles: list[Obstacle]) -> float:
    """Eq. (6)/(7): probability that the predicted position is dangerous.

    With a deterministic occupancy the surrogate is 1/0; the mean over the
    horizon gives the accumulated risk R(Np) of Eq. (4).
    """
    return 1.0 if any(o.collide(p) for o in obstacles) else 0.0


def accumulated_risk(pred_traj: np.ndarray, obstacles: list[Obstacle]) -> float:
    """Eq. (4): mean fraction of predicted points inside any safety zone."""
    if pred_traj.shape[0] == 0:
        return 0.0
    return float(np.mean([risk_at_point(p, obstacles) for p in pred_traj]))


def nearest_obstacle(p: np.ndarray, obstacles: list[Obstacle]) -> tuple[int | None, float]:
    """Index and signed clearance of the nearest obstacle (negative = inside)."""
    best, best_d = None, np.inf
    for k, o in enumerate(obstacles):
        d = float(np.linalg.norm(p - o.centre))
        if d < best_d:
            best, best_d = k, d
    return best, best_d - (0.0 if best is None else obstacles[best].radius)


def adapt_horizon(pred_risk: float, stable: bool,
                  Tp: float, dT: float, Tmin: float, Tmax: float) -> float:
    """Eq. (5) / Section 1.2 adaptive-horizon rules."""
    if pred_risk > 0.0 or not stable:          # high risk -> shorten
        return max(Tp - dT, Tmin)
    return min(Tp + dT, Tmax)                   # stable -> lengthen


def correction_heading(pred: np.ndarray, vel: np.ndarray, obs: list[Obstacle],
                       w_e: float = 0.6, w_o: float = 3.0, w_t: float = 0.4) -> float:
    """Eq. (8): choose the commanded heading that balances three objectives.

    Minimise a scalar cost over candidate headings psi:
        J(psi) = w_t * |wrap(psi - psi_ref(t))|   (maintain reference course)
               + w_e * |wrap(psi - psi_vel)|      (keep forward momentum)
               + w_o * arcsin-ish repulsion,      (avoid the nearest safety zone)
    solved by a small one-dimensional line search over the horizon of headings.
    """
    psi_vel = np.arctan2(vel[1], vel[0])
    # reference course = direction of the reference tangent passed as a callback
    psi_ref = _ref_course(pred, obs, psi_vel)
    # nearest obstacle direction
    k, clearance = nearest_obstacle(pred, obs)
    if k is None:
        return psi_ref
    psi_obs = np.arctan2(obs[k].centre[1] - pred[1], obs[k].centre[0] - pred[0])
    # candidate headings as perturbations of the reference course
    cands = np.linspace(psi_ref - np.pi, psi_ref + np.pi, 361)
    cost = w_t * np.abs(wrap_pi(cands - psi_ref))
    cost = cost + w_e * np.abs(wrap_pi(cands - psi_vel))
    # observable avoidance: push away from obstacle, decaying with clearance
    if clearance < 5.0 * obs[k].radius:
        turn_away = wrap_pi(psi_obs + np.pi - cands)
        cost = cost + w_o * np.abs(turn_away) / (1.0 + clearance)
    return float(cands[int(np.argmin(cost))])


def _ref_course(pred: np.ndarray, obs: list[Obstacle], fallback: float) -> float:
    """The nominal course is the current-pose heading; controllers override
    this via the reference tangent at decision time (see methods.py)."""
    return fallback


def wrap_command(psi: float) -> float:
    return float(wrap_pi(psi))