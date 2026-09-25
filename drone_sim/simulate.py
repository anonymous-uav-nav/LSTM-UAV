"""Trajectory-following simulation + metric collection.

Runs one UAV along a reference `Path` under a `Method`, applying:
  * GPS measurement noise (N(0,sigma), the +/-3 m clue of Section 1.4),
  * a yaw-rate limit |dpsi| <= wdot_max*dt (finite agility -> the Eq. (3) bound),
  * obstacle safety zones maintained in the avoidance modules.

Collects the Section 2 metrics: mean deviation, adaptation time, number of
evasion maneuvers, minimum obstacle distance, mean energy, success rate and
the reconstructed OPE.
"""
from __future__ import annotations

import numpy as np
from .trajectories import Path
from .avoidance import Obstacle, nearest_obstacle
from .energy import energy_metrics


@np.errstate(divide="ignore", invalid="ignore")
def simulate(method, path: Path, obstacles: list[Obstacle],
             steps: int = 400, dt: float = 0.5, v: float = 6.0,
             sigma_gps: float = 1.5, wdot_max: float = 0.35,
             dev_threshold: float = 12.0, seed: int = 0,
             warmup: int = 15,
             wind: tuple = (0.0, 0.0), gust_sigma: float = 0.0) -> dict:
    """Run the closed loop; return the metric dictionary.

    Parameters
    ----------
    dev_threshold : the 'permissible deviation' that defines the success rate
        (the paper leaves this value implicit; 12 m is the stated default).
    warmup : the LSTM needs a priming window -> first `warmup` steps are steered
        by pure pursuit so prediction starts from a valid feature buffer.
    wind : constant mean wind vector (m/s) disturbing the TRUE position.
    gust_sigma : std of the turbulence added on top of the mean wind
        (discrete Dryden-lite: white noise scaled by sqrt(dt)).
    """
    rng = np.random.default_rng(seed)
    # start on the path
    s_start = 0.0
    pos = np.array(path.pos(s_start), dtype=float)
    psi = float(np.arctan2(path.tangent(s_start)[1], path.tangent(s_start)[0]))

    true_pos, meas_pos = [pos.copy()], [pos.copy()]
    heading_log = [psi]
    cleared = []                       # min obstacle clearance per step
    evad_log = []                      # in a risking state per step

    for t in range(steps):
        # noisy measurement
        meas = pos + rng.normal(0, sigma_gps, 2)
        vel = v * np.array([np.cos(psi), np.sin(psi)])
        state = {
            "pos": meas, "vel": vel, "psi": psi,
            "path": path, "obstacles": obstacles, "dt": dt,
        }
        psi_cmd, evading = method.step(state)

        # yaw-rate saturation (finite agility)
        dpsi = np.arctan2(np.sin(psi_cmd - psi), np.cos(psi_cmd - psi))
        dpsi = np.clip(dpsi, -wdot_max * dt, wdot_max * dt)
        psi += dpsi

        # integrate (wind + optional turbulence act on the TRUE position only;
        # the controller still only sees noisy GPS, as in the base setting)
        pos = pos + v * dt * np.array([np.cos(psi), np.sin(psi)]) \
            + np.asarray(wind) * dt
        if gust_sigma > 0.0:
            pos = pos + rng.normal(0.0, gust_sigma * np.sqrt(dt), 2)
        true_pos.append(pos.copy())
        meas_pos.append(meas.copy())
        heading_log.append(psi)
        evad_log.append(evading)

        # min clearance for safety metric
        _k, cl = nearest_obstacle(pos, obstacles)
        cleared.append(cl)

    true_pos = np.array(true_pos)
    meas_pos = np.array(meas_pos)

    # ------------------------------ metrics ------------------------------
    # 1. mean deviation from the reference
    devs, s_proj = [], []
    for p in true_pos:
        _s, cross, _ = path.closest(p)
        devs.append(abs(cross))
        s_proj.append(_s)
    devs = np.array(devs); s_proj = np.array(s_proj)
    mean_dev = float(np.mean(devs[0:]))                      # (m)

    # 2. adaptation time after a deviation (time to re-enter dev_threshold)
    adapt = None
    recent_off = devs > dev_threshold
    i = 0
    while i < len(devs):
        if recent_off[i]:
            start = i
            while i < len(devs) and recent_off[i]:
                i += 1
            # found an episode of being off -> measure return time (in seconds)
            if i < len(devs) - 1 and adapt is None:
                adapt = float((i - start) * dt)
            break
        i += 1
    adaptation = adapt if adapt is not None else 0.0

    # 3. number of evasion maneuvers (rising edges of the risky state)
    evad_arr = np.array(evad_log, dtype=int)
    n_evade = int(np.sum(np.diff(np.concatenate([[0], evad_arr])) > 0))

    # 4. minimum distance to obstacle (>= safety radius => free)
    min_dist = float(np.min(cleared)) + min(o.radius for o in obstacles)

    # 5. mean energy (heading-based, Section 1.4)
    E = energy_metrics(true_pos)
    mean_e = E["E_mean"]

    # 6. success rate: fraction of samples inside dev_threshold
    success = float(np.mean(devs <= dev_threshold)) * 100.0

    return {
        "mean_dev": mean_dev,
        "adaptation": adaptation,
        "n_evade": n_evade,
        "min_dist": min_dist,
        "mean_energy": mean_e,
        "success": success,
        # NOTE: n_steps must count the energy samples (E.size = steps - 1),
        # NOT the simulation step count, so that the Table-1 identity
        # E_total == E_mean * n_steps holds exactly.
        "n_steps": int(E["n_steps"]),
        "traj": true_pos,
        "meas": meas_pos,
        "proj_s": s_proj,
        "energy_total": E["E_total"],
        "energy_sigma": E["E_sigma"],
    }


def batch_ope(results: dict[str, dict]) -> dict[str, float]:
    """Empirically normalised Overall Performance Efficiency (higher is better).

    Each of the five metrics is min-max normalised over the compared batch to
    [0, 1] with the *good direction* (lower deviation/evasion/energy and higher
    success map to 1). OPE is their equal-weight mean, so a coherent 0-1
    "productivity coefficient" emerges without relying on unpublished weights.
    """
    def norm(key, better_low):
        vals = {n: r[key] for n, r in results.items()}
        lo, hi = min(vals.values()), max(vals.values())
        span = (hi - lo) or 1.0
        out = {}
        for n, r in results.items():
            z = (r[key] - lo) / span
            out[n] = 1.0 - z if better_low else z
        return out
    keys = [("mean_dev", True), ("mean_energy", True), ("n_evade", True),
            ("success", False), ("min_dist", False)]
    nal = {k: norm(k, lo) for k, lo in keys}
    ope = {}
    for n in results:
        ope[n] = float(np.mean([nal[k][n] for k, _ in keys]))
    return ope