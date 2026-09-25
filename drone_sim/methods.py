"""The five controllers compared in Section 2.

Common interface: a `Method` exposes

    step(state: dict) -> (psi_cmd: float, evading: bool)

`state` carries the noisy measurement, the reference `Path`, the obstacle list
and `dt`; the AI method additionally holds the trained LSTM predictor.

Evasion accounting: every method reports `evading = True` for the step in which
it decides to deflect (reactive tangential fly-around for the baselines; a
predicted safety-zone conflict for the AI). The simulator counts rising edges,
which is how "number of obstacle avoidance maneuvers" is tallied uniformly.
"""
from __future__ import annotations

import numpy as np
from .trajectories import Path
from .avoidance import Obstacle, correction_heading, accumulated_risk, nearest_obstacle


def _reference_course(state) -> float:
    s0, _cross, _ = state["path"].closest(state["pos"])
    tan = state["path"].tangent(s0)
    return float(np.arctan2(tan[1], tan[0]))


def _reactive_avoid(state, psi_nominal: float, danger: float = 40.0):
    """Tangential fly-around when an obstacle is within `danger` metres.

    Returns (heading, evading). A consistent turn sense is chosen by the sign of
    the cross product so the UAV does not weave between two symmetric threats.
    """
    pos = state["pos"]
    for o in state["obstacles"]:
        rel = o.centre - pos
        d = float(np.linalg.norm(rel))
        if d < danger and d > 1e-6:
            psi_obs = np.arctan2(rel[1], rel[0])
            t = np.array([np.cos(psi_nominal), np.sin(psi_nominal)])
            sign = 1.0 if (np.cross(t, rel) > 0) else -1.0
            return float(psi_obs + sign * np.pi / 2), True
    return float(psi_nominal), False


class PurePursuit:
    name = "Pure Pursuit"

    def __init__(self, lookahead: float = 30.0):
        self.L = lookahead

    def step(self, state):
        s0, _, _ = state["path"].closest(state["pos"])
        s_aim = min(s0 + self.L, state["path"].total_length())
        px, py = state["path"].pos(s_aim)
        psi = np.arctan2(py - state["pos"][1], px - state["pos"][0])
        psi, evad = _reactive_avoid(state, psi)
        return psi, evad


class LineOfSight:
    name = "Line of Sight"

    def __init__(self, lookahead: float = 30.0, k: float = 0.5):
        self.L, self.k = lookahead, k

    def step(self, state):
        s0, e, _ = state["path"].closest(state["pos"])
        psi_ref = _reference_course(state)
        los = psi_ref + np.arctan(-self.k * e / (self.L + 1e-9))
        psi, evad = _reactive_avoid(state, float(los))
        return psi, evad


class VectorField:
    """Asymptotic guidance field onto the path (bounded cross-track gain)."""

    name = "Vector Field"

    def __init__(self, k: float = 0.5, U: float = 6.0):
        self.k, self.U = k, U

    def step(self, state):
        s0, e, _ = state["path"].closest(state["pos"])
        tan = state["path"].tangent(s0)
        # guidance law: head along the tangent, pulled onto it by a bounded gain
        angle = np.arctan2(tan[1], tan[0]) - np.arctan(self.k * e / self.U)
        field = np.array([np.cos(angle), np.sin(angle)])
        # repulsion is delegated to the shared tangential avoid (keeps it stable)
        psi = float(np.arctan2(field[1], field[0]))
        psi, evad = _reactive_avoid(state, psi, danger=40.0)
        return psi, evad


class NonlinearStabilization:
    name = "Nonlinear Stabilization"

    def __init__(self, k: float = 0.06):
        self.k = k

    def step(self, state):
        s0, e, _ = state["path"].closest(state["pos"])
        psi_ref = _reference_course(state)
        psi = psi_ref + np.arctan(-self.k * e)
        psi, evad = _reactive_avoid(state, float(psi))
        return psi, evad


class ProposedAI:
    """LSTM-aided kinematic filter + predictive pursuit + proactive avoidance.

    The recurrent net predicts the *clean step increment* from a window of noisy
    velocity/heading/delta features (Section 1.2). That prediction drives three
    parts of the loop (Sections 1.3-1.4):

      * state estimate: p_hat = beta*(p_hat + dz) + (1-beta)meas  - a learned,
        measurement-anchored filter whose prior comes from the LSTM, so the
        commanded track is smooth (low heading jitter -> low manoeuvring energy);
      * risk: extrapolating p_hat forward over `horizon` detects if the forecast
        will enter a safety zone, triggering Eq. (8) *before* arrival;
      * lead: the aim point on the reference path is anchored to a projection of
        p_hat one horizon ahead, cancelling the corner latency reactive methods
        suffer.

    In short: smoothing and "seeing ahead" - the exact two things the paper
    ascribes to the LSTM - are what give the predictive approach its edge,
    without asking a tiny numpy net to regress absolute 300 m coordinates.
    """
    name = "Proposed AI"

    def __init__(self, lstm, window: int = 10, horizon: int = 4,
                 lookahead: float = 25.0, beta: float = 0.35,
                 use_lstm: bool = True, clip_bound: float = 1.5):
        self.lstm = lstm
        self.window, self.horizon, self.L, self.beta = window, horizon, lookahead, beta
        self.use_lstm, self.clip_bound = use_lstm, clip_bound
        self._limit = 8.0                  # max accepted step increment (m)
        self._feat = []                    # [v, cos psi, sin psi, dx_noisy, dy_noisy]
        self._p = None                   # filtered state estimate
        self._prev = None                # previous measurement (for dx_noisy)

    def _push(self, state):
        me = state["pos"].astype(float)
        v = float(np.linalg.norm(state.get("vel", np.array([0.0, 0.0]))))
        psi = float(state.get("psi", 0.0))
        dx = 0.0 if self._prev is None else float(me[0] - self._prev[0])
        dy = 0.0 if self._prev is None else float(me[1] - self._prev[1])
        self._prev = me.copy()
        self._feat.append([v, np.cos(psi), np.sin(psi), dx, dy])
        if len(self._feat) > self.window:
            self._feat.pop(0)
        if self._p is None:
            self._p = me.copy()

    def step(self, state):
        self._push(state)
        me = state["pos"]
        vel = state.get("vel", np.zeros(2))
        if len(self._feat) < self.window:                          # warm-up
            psi, evad = _reactive_avoid(state, _pp_heading(me, state, self.L))
            return psi, evad
        X = np.array(self._feat[-self.window:])
        v = float(np.linalg.norm(vel))
        psi = float(state.get("psi", 0.0))
        dz_kin = v * state["dt"] * np.array([np.cos(psi), np.sin(psi)])   # kinematic motion model
        dz_lstm = self.lstm.smooth(X) if self.use_lstm else dz_kin          # learned correction
        # final prior = kinematics + a *bounded* learned tweak, so a weak model
        # cannot drag the vehicle off-track (clip_bound is the ablation switch)
        dz = dz_kin + np.clip(dz_lstm - dz_kin, -self.clip_bound, self.clip_bound)
        # measurement-anchored filter
        self._p = self.beta * (self._p + dz) + (1 - self.beta) * me
        p = self._p
        # (1) proactive risk from extrapolated filtered track (Eq. 4 -> Eq. 8)
        pred = np.vstack([p + dz * (i + 1) for i in range(self.horizon)]) \
            if self.horizon > 0 else np.zeros((0, 2))
        if accumulated_risk(pred, state["obstacles"]) > 0.0:
            psi = correction_heading(p, vel, state["obstacles"], w_t=0.5, w_e=0.3, w_o=3.0)
            return float(psi), True
        # (2) smooth predictive pursuit: aim at the path ahead of the filtered state
        _k, cl = nearest_obstacle(p, state["obstacles"]) if state["obstacles"] else (None, 1e9)
        look = self.L * (0.6 if cl < self.L else 1.0)
        s_est, _cross, _ = state["path"].closest(p)
        s_aim = min(s_est + look, state["path"].total_length())
        px, py = state["path"].pos(s_aim)
        psi = np.arctan2(py - me[1], px - me[0])
        psi, evad = _reactive_avoid(state, psi, danger=40.0)       # hard near-field safety net
        return psi, evad


def _pp_heading(pos, state, lookahead: float) -> float:
    s0, _, _ = state["path"].closest(pos)
    s_aim = min(s0 + lookahead, state["path"].total_length())
    px, py = state["path"].pos(s_aim)
    return float(np.arctan2(py - pos[1], px - pos[0]))


ALL_METHODS = [PurePursuit, LineOfSight, VectorField, NonlinearStabilization, ProposedAI]