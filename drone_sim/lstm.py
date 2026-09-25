"""A lightweight, dependency-free LSTM for trajectory smoothing and prediction.

Two uses, matching the paper's Section 1.2-1.4:

1. Sequence regression, Eq. (1): given a window of features, predict each next
   absolute position (teacher-forced MSE over the window).
   `train_sequence(...)`, `predict_seq(...)`.

2. Recurrent smoothing: given a *noisy* recent window, predict the *clean*
   current position (the paper's "LSTM smooths GPS errors"). This is a
   well-conditioned regression (errors ~ GPS noise, not ~ path scale) and is
   what the closed-loop controller uses to denoise the measurement.
   `train_smoother(...)`, `smooth(...)`.

Written in raw numpy / BPTT so the reconstruction needs only `numpy` + `pandas`
- the exact stack the paper names - and calls no deep-learning framework.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass
class LSTM:
    n_input: int = 4          # features per step
    n_hidden: int = 16        # hidden size
    n_out: int = 2            # (x, y)
    scale: float = 100.0      # metres -> normalised coordinate scaling
    seed: int = 7
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        D, H, O = self.n_input, self.n_hidden, self.n_out
        self.Wx = self.rng.normal(0, 1 / np.sqrt(D), (4 * H, D))
        self.Wh = self.rng.normal(0, 1 / np.sqrt(H), (4 * H, H))
        self.b = np.zeros(4 * H)
        self.Wo = self.rng.normal(0, 1 / np.sqrt(H), (O, H))
        self.bo = np.zeros(O)

    def gates_pre(self, x: np.ndarray, h_prev: np.ndarray | None) -> np.ndarray:
        return self.Wx @ x + (self.Wh @ h_prev if h_prev is not None else 0.0) + self.b

    def forward(self, X: np.ndarray, h0=None, c0=None):
        """X: (T, D). Returns out (T, O), h (T, H), c (T, H)."""
        T = X.shape[0]; H = self.n_hidden
        out, hs, cs = np.zeros((T, self.n_out)), np.zeros((T, H)), np.zeros((T, H))
        h = np.zeros(H) if h0 is None else h0
        c = np.zeros(H) if c0 is None else c0
        for t in range(T):
            a = self.gates_pre(X[t], h)
            i, f, o, g = np.split(a, 4)
            i, f, o, g = _sigmoid(i), _sigmoid(f), _sigmoid(o), np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            out[t] = self.Wo @ h + self.bo
            hs[t], cs[t] = h, c
        return out, hs, cs

    # ---------------------------------------------------------------- forward
    def smooth(self, X: np.ndarray) -> np.ndarray:
        """Clean current-position estimate (metres) for a (T, D) noisy window."""
        out, _, _ = self.forward(np.asarray(X, dtype=float))
        return out[-1] * self.scale

    def predict_seq(self, X: np.ndarray, steps: int) -> np.ndarray:
        """Recursive multi-step forecast; returns (steps, 2) metres.

        Feeds predicted points back into the window (constant speed/heading for
        the closed loop) - the error accumulation of Eq. (2) is real.
        """
        feats = [list(r) for r in np.asarray(X, dtype=float)]
        out = np.zeros((steps, 2)); h = c = np.zeros(self.n_hidden)
        for k in range(steps):
            _, hs, cs = self.forward(np.array(feats, dtype=float), h, c)
            h, c = hs[-1], cs[-1]
            px, py = self.Wo @ h + self.bo
            out[k] = (px * self.scale, py * self.scale)
            feats.append([px, py, feats[-1][2], feats[-1][3]])
            feats.pop(0)
        return out

    # ---------------------------------------------------------- BPTT helpers
    def _bptt(self, X: np.ndarray, dLdout: np.ndarray):
        """Full accumulated gradients for one window.

        `_bptt_single` returns (gWx, gWh, gb, gWo, gbo) in one forward+backward
        pass (hidden states are computed once and reused).
        """
        H = self.n_hidden
        _, hs, cs = self.forward(X)
        gWx = np.zeros_like(self.Wx); gWh = np.zeros_like(self.Wh)
        gb = np.zeros_like(self.b); gWo = np.zeros_like(self.Wo); gbo = np.zeros_like(self.bo)
        dc_next = np.zeros(H); dh_carry = np.zeros(H)
        for t in range(len(X) - 1, -1, -1):
            dLdh = self.Wo.T @ dLdout[t] + dh_carry
            lh = hs[t - 1] if t > 0 else None
            a = self.gates_pre(X[t], lh)
            i, f, o = _sigmoid(a[:H]), _sigmoid(a[H:2 * H]), _sigmoid(a[2 * H:3 * H])
            g = np.tanh(a[3 * H:])
            c_t = cs[t]; tanhc = np.tanh(c_t)
            dc_full = dc_next + dLdh * o * (1 - tanhc ** 2)
            raw = np.concatenate([dc_full * g * i * (1 - i),                 # di
                                  (dc_full * (cs[t - 1] if t > 0 else 0.0)) * f * (1 - f),
                                  dLdh * tanhc * o * (1 - o),                # do
                                  dc_full * i * (1 - g ** 2)])               # dg
            gWx += raw[:, None] @ X[t][None, :]
            gb += raw
            if t > 0:
                gWh += raw[:, None] @ hs[t - 1][None, :]
                dh_carry = self.Wh.T @ raw
            dc_next = dc_full * f
            if np.any(dLdout[t]):                                # output layer
                gWo += dLdout[t][:, None] * hs[t][None, :]
                gbo += dLdout[t]
        return gWx, gWh, gb, gWo, gbo

    # -------------------------------------------------------------- training
    def train_sequence(self, Xs, Ys, epochs=60, lr=0.05, clip=5.0):
        """Eq. (1): teacher-forced MSE over the window; Ys (B, T-1, 2) *scaled*.
        Returns per-epoch normalised MSE (multiply by scale**2 for m^2)."""
        return self._train(Xs, Ys, epochs, lr, clip, smo=False)

    def train_smoother(self, Xs, Ys, epochs=60, lr=0.05, clip=5.0):
        """Map noisy window -> clean current position; Ys (B,2) *scaled*."""
        return self._train(Xs, Ys, epochs, lr, clip, smo=True)

    def _train(self, Xs, Ys, epochs, lr, clip, smo):
        losses = []
        for ep in range(epochs):
            if ep % 5 == 4:
                lr *= 0.7
            tot = 0.0
            for X, Y in zip(Xs, Ys):
                out, _, _ = self.forward(X)
                if smo:
                    dLdout = np.zeros_like(out)
                    dLdout[-1] = out[-1] - Y
                    tot += float(np.mean(dLdout[-1] ** 2))
                else:
                    T = X.shape[0]
                    dLdout = np.zeros_like(out)
                    dLdout[:-1] = (out[:-1] - Y) / max(1, T - 1)
                    tot += float(np.mean((out[:-1] - Y) ** 2))
                grads = self._bptt(X, dLdout)
                for grads_i, par in zip(grads, (self.Wx, self.Wh, self.b, self.Wo, self.bo)):
                    np.clip(grads_i, -clip, clip, out=grads_i)
                    par -= lr * grads_i / max(1, len(Xs))
            losses.append(tot / max(1, len(Xs)))
        return losses


def make_delta_windows(meas: np.ndarray, truth: np.ndarray, vel: np.ndarray,
                       heading: np.ndarray, window: int):
    """Features for the *increment* predictor (for the kinematic filter).

    Features are dimensionless / O(1): [v, cos psi, sin psi, dx_noisy, dy_noisy]
    where the noisy step is the difference of consecutive *noisy* measurements
    the controller sees. Target Y is the *clean* step increment at the window's
    end. `n_input` for such a net is 5 and `scale` should be 1.0 (meters).
    """
    d = np.vstack([[0.0, 0.0], np.diff(meas, axis=0)])   # noisy step increments
    c, s = np.cos(heading), np.sin(heading)
    Xs, Ys = [], []
    for i in range(0, len(meas) - window - 1):
        feats = np.stack([vel[i:i + window], c[i:i + window], s[i:i + window],
                          d[i:i + window, 0], d[i:i + window, 1]], axis=1)
        Xs.append(feats)
        Ys.append(truth[i + window] - truth[i + window - 1])   # clean step
    return np.array(Xs), np.array(Ys)


def make_windows(meas: np.ndarray, truth: np.ndarray, vel: np.ndarray, heading: np.ndarray,
                 window: int, scale: float = 100.0, smoother: bool = True):
    """Build (X, Y) pairs.

    Features are [x/scale, y/scale, speed, sin(heading)] from *meas* (the noisy
    trace the controller sees). With `smoother=True`, Y is the *true* absolute
    position at each window's last index (clean-current regression). Otherwise
    Y is absolute next positions along the window (Eq. 1), length window-1.
    """
    D = np.stack([meas[:, 0] / scale, meas[:, 1] / scale, vel, np.sin(heading)], axis=1)
    Xs, Ys = [], []
    for s in range(0, len(meas) - window):
        Xs.append(D[s:s + window])
        if smoother:
            Ys.append(truth[s + window - 1] / scale)
        else:
            Ys.append(meas[s + 1:s + window] / scale)
    return np.array(Xs), np.array(Ys)