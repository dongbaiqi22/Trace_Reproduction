from __future__ import annotations

import math
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional


# -----------------------------
# Utilities
# -----------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def sample_positions_in_disk(n: int, radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0),
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Uniformly sample n points in a disk of given radius and center.
    Returns array of shape (n, 2).
    """
    rng = _rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def rotate_points(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate 2D points by angle_rad (counter-clockwise).
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s,  c]], dtype=xy.dtype)
    return xy @ R.T


def assign_types_half_disk(xy: np.ndarray, boundary_angle_rad: float = 0.0,
                           cross_boundary_band: float = 0.0,
                           band_flip_prob: float = 0.0,
                           global_flip_frac: float = 0.0,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Assign two types by splitting the disk with a rotated diameter line.
    - boundary_angle_rad: angle of the diameter normal (0 -> vertical split along y-axis; left vs right).
    - cross_boundary_band: half-width of a band around the boundary (in absolute x_rot units
      normalized by the disk radius ~1.0 if you sampled in unit circle). Points within this band
      can be flipped with probability band_flip_prob.
    - global_flip_frac: additional random flips anywhere to create cross-boundary noise.
    Returns labels array of shape (n,) with values in {0,1}.
    """
    rng = _rng(seed)
    xy_rot = rotate_points(xy, -boundary_angle_rad)
    x_rot = xy_rot[:, 0]

    labels = (x_rot >= 0).astype(np.int64)

    if cross_boundary_band > 0.0 and band_flip_prob > 0.0:
        in_band = np.abs(x_rot) <= cross_boundary_band
        flip_mask = in_band & (rng.uniform(0, 1, size=xy.shape[0]) < band_flip_prob)
        labels[flip_mask] = 1 - labels[flip_mask]

    if global_flip_frac > 0.0:
        n = xy.shape[0]
        k = int(round(global_flip_frac * n))
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            labels[idx] = 1 - labels[idx]

    return labels


# -----------------------------
# Time-series generation
# -----------------------------

def ar1_noise(T: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate AR(1) noise of length T: x_t = phi * x_{t-1} + eps_t, eps ~ N(0, sigma^2).
    Returns array shape (T,).
    """
    x = np.empty(T, dtype=np.float32)
    x[0] = rng.normal(loc=0.0, scale=sigma)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal(loc=0.0, scale=sigma)
    return x


def zscore(a: np.ndarray, axis: Optional[int] = None, eps: float = 1e-8) -> np.ndarray:
    m = np.mean(a, axis=axis, keepdims=True)
    s = np.std(a, axis=axis, keepdims=True)
    return (a - m) / (s + eps)


def default_type_waveforms(T: int, baseline_frac: float = 0.8, amp: float = 1.0,
                           rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create two type-specific shared waveforms (u_A, u_B) with a short informative window.
    - First baseline_frac of time is near-zero (uninformative), final window carries opposite polarity.
    - Waveforms are z-scored to var=1.
    Returns (u_A, u_B), each shape (T,).
    """
    if rng is None:
        rng = _rng(None)
    T_b = int(round(baseline_frac * T))
    u = np.zeros(T, dtype=np.float32)

    window = max(8, T - T_b)
    t = np.arange(window, dtype=np.float32)
    # half-sine bump then exponential decay to mimic transient
    bump = np.sin(np.pi * (t / window))
    decay = np.exp(-t / max(4.0, window / 6.0)).astype(np.float32)
    w = bump * decay
    w = w / (np.max(np.abs(w)) + 1e-8) * amp

    u[T_b:T_b + window] = w[: max(0, min(window, T - T_b))]

    # small random smooth component so the two types aren't perfectly anti-symmetric
    jitter = ar1_noise(T, phi=0.9, sigma=0.05, rng=rng)
    uA = zscore(u + 0.5 * jitter).astype(np.float32)
    uB = zscore(-0.9 * u + 0.5 * jitter).astype(np.float32)
    return uA, uB


def calcium_kernel(T: int, dt: float, tau: float) -> np.ndarray:
    """
    Exponential calcium decay kernel of length T: k[t] = exp(-t*dt/tau). Not normalized.
    """
    t = np.arange(T, dtype=np.float32)
    k = np.exp(-t * dt / max(tau, 1e-3)).astype(np.float32)
    return k


def causal_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal 1D convolution (valid length equals original signal length) using 'full' then trim.
    signal: shape (..., T)
    kernel: shape (K,)
    Returns: array with same trailing length T.
    """
    T = signal.shape[-1]
    full = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="full")[:T], -1, signal)
    return full


@dataclass
class SimulatorConfig:
    n_neurons: int = 500
    n_trials: int = 12
    T: int = 300
    dt: float = 1/8
    radius: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    boundary_angle_rad: float = 0.0
    cross_boundary_band: float = 0.05
    band_flip_prob: float = 0.5
    global_flip_frac: float = 0.0

    target_within_type_corr: float = 0.2
    alpha_std: float = 0.10
    baseline_frac: float = 0.8
    amp: float = 1.0
    neuron_noise_phi: float = 0.6
    trial_noise_phi: float = 0.4
    neuron_noise_std: float = 1.0
    trial_noise_std: float = 0.3
    meas_noise_std: float = 0.05

    tau_decay: float = 0.8
    apply_calcium: bool = True

    add_artifact: bool = False
    artifact_rect: Tuple[float, float, float, float] = (-1.0, -1.0, -0.2, 1.0)
    artifact_scale: float = 0.8

    # Random seed
    seed: Optional[int] = 42


def generate_dataset(cfg: SimulatorConfig) -> Dict[str, np.ndarray]:
    """
    Generate a spatially-structured, multi-trial dataset.
    Returns a dict with keys:
      - positions: (N,2)
      - labels: (N,) in {0,1}
      - traces: (N, n_trials, T) float32
      - uA, uB: (T,) shared waveforms per type
      - alphas: (N,) neuron mixing coefficients
      - params: a small dict of config values for reproducibility
    """
    rng = _rng(cfg.seed)

    positions = sample_positions_in_disk(cfg.n_neurons, radius=cfg.radius,
                                         center=(cfg.center_x, cfg.center_y), seed=rng.integers(1<<31))
    labels = assign_types_half_disk(
        positions,
        boundary_angle_rad=cfg.boundary_angle_rad,
        cross_boundary_band=cfg.cross_boundary_band,
        band_flip_prob=cfg.band_flip_prob,
        global_flip_frac=cfg.global_flip_frac,
        seed=rng.integers(1<<31),
    )

    uA, uB = default_type_waveforms(cfg.T, baseline_frac=cfg.baseline_frac, amp=cfg.amp, rng=rng)

    mu_alpha = float(np.sqrt(max(cfg.target_within_type_corr, 1e-6)))
    alphas = np.clip(rng.normal(loc=mu_alpha, scale=cfg.alpha_std, size=cfg.n_neurons), 0.0, 1.0).astype(np.float32)

    eps = np.stack([zscore(ar1_noise(cfg.T, cfg.neuron_noise_phi, cfg.neuron_noise_std, rng)) for _ in range(cfg.n_neurons)], axis=0).astype(np.float32)

    base = np.empty((cfg.n_neurons, cfg.T), dtype=np.float32)
    for i in range(cfg.n_neurons):
        u = uA if labels[i] == 0 else uB
        ai = alphas[i]
        base[i] = ai * u + np.sqrt(max(1.0 - ai * ai, 0.0)) * eps[i]

    base = zscore(base, axis=1).astype(np.float32)

    if cfg.apply_calcium:
        k = calcium_kernel(cfg.T, cfg.dt, cfg.tau_decay)
    else:
        k = np.array([1.0], dtype=np.float32)

    traces = np.empty((cfg.n_neurons, cfg.n_trials, cfg.T), dtype=np.float32)
    for l in range(cfg.n_trials):
        trial_eps = np.stack([ar1_noise(cfg.T, cfg.trial_noise_phi, cfg.trial_noise_std, rng) for _ in range(cfg.n_neurons)], axis=0).astype(np.float32)
        trial_signal = base + trial_eps

        if cfg.add_artifact:
            xmin, ymin, xmax, ymax = cfg.artifact_rect
            in_rect = (positions[:, 0] >= xmin) & (positions[:, 0] <= xmax) & (positions[:, 1] >= ymin) & (positions[:, 1] <= ymax)
            stim = np.abs(uA)[None, :]
            trial_signal[in_rect] = trial_signal[in_rect] + cfg.artifact_scale * stim


        convolved = causal_convolve(trial_signal, k)

        meas = _rng(rng.integers(1<<31)).normal(0.0, cfg.meas_noise_std, size=convolved.shape).astype(np.float32)

        traces[:, l, :] = convolved + meas

    params = asdict(cfg)
    return dict(positions=positions.astype(np.float32),
                labels=labels.astype(np.int64),
                traces=traces.astype(np.float32),
                uA=uA.astype(np.float32),
                uB=uB.astype(np.float32),
                alphas=alphas.astype(np.float32),
                params=params)


def estimate_pairwise_corr_summary(traces: np.ndarray, labels: np.ndarray, n_pairs: int = 5000,
                                   seed: Optional[int] = 0) -> Dict[str, float]:
    """
    Estimate mean pairwise correlations within type-0, within type-1, and between types,
    using the per-neuron mean across trials.
    Computes correlations on flattened time series (length T), sampling up to n_pairs pairs per group.
    Returns a small summary dict.
    """
    rng = _rng(seed)
    x = traces.mean(axis=1)  # (N,T)
    x = zscore(x, axis=1)

    def sample_pairs(idx: np.ndarray) -> float:
        if len(idx) < 2:
            return float("nan")
        m = len(idx)
        n = min(n_pairs, m * (m - 1) // 2)
        vals = []
        for _ in range(n):
            i, j = rng.choice(idx, size=2, replace=False)
            ci = np.corrcoef(x[i], x[j])[0, 1]
            vals.append(ci)
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    within0 = sample_pairs(idx0)
    within1 = sample_pairs(idx1)

    def sample_between(i0: np.ndarray, i1: np.ndarray) -> float:
        if len(i0) == 0 or len(i1) == 0:
            return float("nan")
        n = min(n_pairs, len(i0) * len(i1))
        vals = []
        for _ in range(n):
            i = rng.choice(i0)
            j = rng.choice(i1)
            ci = np.corrcoef(x[i], x[j])[0, 1]
            vals.append(ci)
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    between = sample_between(idx0, idx1)

    return {"within_type0_mean_corr": within0,
            "within_type1_mean_corr": within1,
            "between_types_mean_corr": between}


if __name__ == "__main__":
    cfg = SimulatorConfig(
        n_neurons=1000, n_trials=10, T=300, seed=123,
        boundary_angle_rad=0.0,
        cross_boundary_band=0.05, band_flip_prob=0.5,
        global_flip_frac=0.02,
        target_within_type_corr=0.2, alpha_std=0.10,
        add_artifact=True
    )
    data = generate_dataset(cfg)
    summary = estimate_pairwise_corr_summary(data["traces"], data["labels"], n_pairs=2000, seed=0)
    print("Config summary:", json.dumps(cfg.__dict__, indent=2))
    print("Positions:", data["positions"].shape, "Traces:", data["traces"].shape)
    print("Corr summary:", summary)
