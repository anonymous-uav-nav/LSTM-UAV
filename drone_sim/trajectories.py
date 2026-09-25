"""Reference trajectories for the UAV mission reconstruction.

Every reference path is represented internally as a *dense polyline sampling*
so that straight, circular, and polygonal routes share one implementation of
closest-point projection (used by the cross-track-deviation and LOS metrics).
Sampling spacing is taken fine enough (0.25 m) that projection error is far
below the GPS noise floor.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

SAMPLE_DX = 0.25  # metres between dense samples of the reference path


@dataclass
class Path:
    """Dense polyline reference path with arclength parametrisation."""

    pts: np.ndarray          # M x 2 sampled reference points
    s: np.ndarray            # M arclength samples (cumulative)

    @classmethod
    def from_polyline(cls, vertices: np.ndarray, step: float = SAMPLE_DX) -> "Path":
        """Build a path by resampling a polyline (given vertex list) at `step`."""
        verts = np.asarray(vertices, dtype=float)
        seg = np.diff(verts, axis=0)                 # (K-1,2)
        lengths = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        n = max(2, int(np.ceil(cum[-1] / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        # locate the polyline segment index for each sample
        s_target = t * cum[-1]
        idx = np.searchsorted(cum, s_target, side="right") - 1
        idx = np.clip(idx, 0, len(verts) - 2)
        seg_local = (s_target - cum[idx])[:, None] / np.maximum(lengths[idx][:, None], 1e-12)
        pts = verts[idx] + seg_local * seg[idx]
        return cls(pts=pts, s=s_target)

    @classmethod
    def straight(cls, a: np.ndarray, b: np.ndarray) -> "Path":
        return cls.from_polyline(np.array([a, b], dtype=float))

    @classmethod
    def circle(cls, centre: np.ndarray, radius: float, segments: int = 100) -> "Path":
        th = np.linspace(0.0, 2 * np.pi, segments)
        pts = centre + radius * np.stack([np.cos(th), np.sin(th)], axis=1)
        return cls.from_polyline(np.vstack([pts, pts[0]]))  # close the ring
        # (the closing edge is a zero-length duplicate, harmless)

    @classmethod
    def polyline(cls, vertices: np.ndarray) -> "Path":
        return cls.from_polyline(np.asarray(vertices, dtype=float))

    def total_length(self) -> float:
        return float(self.s[-1])

    def pos(self, s: float) -> np.ndarray:
        """Position(s) at arclength `s` (clamped to [0, total])."""
        return np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])

    def tangent(self, s: float) -> np.ndarray:
        """Unit tangent at arclength `s` (central differences)."""
        p0 = np.interp(s - 1.0, self.s, self.pts[:, 0]), np.interp(s - 1.0, self.s, self.pts[:, 1])
        p1 = np.interp(s + 1.0, self.s, self.pts[:, 0]), np.interp(s + 1.0, self.s, self.pts[:, 1])
        d = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([1.0, 0.0])

    def closest(self, p: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Nearest arclength `s`, signed cross-track distance `e`, closest point `q`.

        Signed error is positive when `p` lies to the left of the forward tangent.
        """
        d = self.pts - np.asarray(p, dtype=float)      # (M,2)
        dist2 = np.einsum("ij,ij->i", d, d)
        i = int(np.argmin(dist2))
        # signed lateral offset using the tangent at nearest sample
        tan = self.tangent(self.s[i])
        lateral = np.cross(tan, np.array(p) - self.pts[i])
        return float(self.s[i]), float(lateral), self.pts[i].copy()