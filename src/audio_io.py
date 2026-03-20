import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


def load_wav_mono(path: str):
    sr, data = wavfile.read(path)

    data = data.astype(np.float64)

    # Convert integer PCM to float [-1, 1] approximately
    if data.dtype != np.float64:
        data = data.astype(np.float64)

    if data.ndim == 2:
        data = data.mean(axis=1)

    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data = data / max_abs

    return sr, data


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio.copy()
    return audio / peak


def bandpass_filter(audio: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, audio)
