import numpy as np


def difference_function(frame: np.ndarray, max_tau: int) -> np.ndarray:
    """
    YIN difference function:
    d(tau) = sum_j (x_j - x_{j+tau})^2
    """
    frame = np.asarray(frame, dtype=np.float64)
    N = len(frame)
    d = np.zeros(max_tau + 1, dtype=np.float64)

    for tau in range(1, max_tau + 1):
        diff = frame[:-tau] - frame[tau:]
        d[tau] = np.sum(diff * diff)

    return d


def cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    cmnd = np.zeros_like(d)
    cmnd[0] = 1.0
    running_sum = 0.0

    for tau in range(1, len(d)):
        running_sum += d[tau]
        if running_sum == 0:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = d[tau] * tau / running_sum

    return cmnd


def absolute_threshold(cmnd: np.ndarray, threshold: float) -> int | None:
    """
    Return first tau under threshold, then descend to local minimum.
    """
    for tau in range(2, len(cmnd)):
        if cmnd[tau] < threshold:
            while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return tau
    return None


def parabolic_interpolation(values: np.ndarray, tau: int) -> float:
    if tau <= 0 or tau >= len(values) - 1:
        return float(tau)

    s0 = values[tau - 1]
    s1 = values[tau]
    s2 = values[tau + 1]

    denom = 2 * (2 * s1 - s2 - s0)
    if denom == 0:
        return float(tau)

    delta = (s2 - s0) / denom
    return tau + delta


def yin_pitch(frame: np.ndarray, sr: int, fmin: float, fmax: float, trough_threshold: float = 0.1):
    """
    Returns: (freq_hz, confidence)
    confidence here is 1 - cmnd(tau), higher is better.
    """
    frame = frame - np.mean(frame)

    if np.max(np.abs(frame)) < 1e-4:
        return None, 0.0

    tau_min = max(2, int(sr / fmax))
    tau_max = min(len(frame) // 2, int(sr / fmin))

    if tau_max <= tau_min:
        return None, 0.0

    d = difference_function(frame, tau_max)
    cmnd = cumulative_mean_normalized_difference(d)

    tau = absolute_threshold(cmnd[tau_min:], trough_threshold)
    if tau is None:
        local_idx = np.argmin(cmnd[tau_min:tau_max + 1])
        tau = tau_min + local_idx
    else:
        tau = tau_min + tau

    tau_refined = parabolic_interpolation(cmnd, tau)

    if tau_refined <= 0:
        return None, 0.0

    freq = sr / tau_refined
    confidence = max(0.0, 1.0 - float(cmnd[int(round(tau))]))

    if freq < fmin or freq > fmax:
        return None, 0.0

    return freq, confidence


def frame_signal(audio: np.ndarray, frame_size: int, hop_size: int):
    num_frames = 1 + max(0, (len(audio) - frame_size) // hop_size)
    frames = []
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frames.append(audio[start:end])
    return frames


def yin_pitch_track(
    audio: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 256,
    fmin: float = 80.0,
    fmax: float = 1200.0,
    trough_threshold: float = 0.1,
):
    window = np.hanning(frame_size)
    frames = frame_signal(audio, frame_size, hop_size)

    times = []
    freqs = []
    confidences = []

    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            break

        frame = frame * window
        freq, conf = yin_pitch(
            frame,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            trough_threshold=trough_threshold,
        )

        t = (i * hop_size + frame_size / 2) / sr
        times.append(t)
        freqs.append(freq)
        confidences.append(conf)

    return np.array(times), freqs, confidences
