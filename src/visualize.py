import numpy as np
import matplotlib.pyplot as plt

from .music import midi_to_note_name


def plot_waveform(audio, sr, title="Waveform"):
    t = np.arange(len(audio)) / sr
    plt.figure(figsize=(12, 3))
    plt.plot(t, audio, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_pitch_track(times, freqs, confidences=None, title="Pitch Track"):
    y = [f if f is not None else np.nan for f in freqs]

    plt.figure(figsize=(12, 4))
    plt.plot(times, y, marker=".", linestyle="-", linewidth=1)

    if confidences is not None:
        weak_x = [times[i] for i, c in enumerate(confidences) if c < 0.6]
        weak_y = [y[i] for i, c in enumerate(confidences) if c < 0.6]
        if weak_x:
            plt.scatter(weak_x, weak_y, marker="x", label="low confidence")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if confidences is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fretboard_positions(best_path, num_frets=20, title="Fretboard Positions"):
    """
    Draw a simple string x fret heatmap and annotate chosen notes.
    Strings shown from 6 (top) to 1 (bottom).
    """
    board = np.zeros((6, num_frets + 1), dtype=float)

    for i, item in enumerate(best_path):
        row = 6 - item["string"]
        col = item["fret"]
        board[row, col] += 1

    plt.figure(figsize=(14, 4))
    plt.imshow(board, aspect="auto")
    plt.colorbar(label="Hit count")
    plt.title(title)
    plt.xlabel("Fret")
    plt.ylabel("String")

    plt.xticks(range(num_frets + 1))
    plt.yticks(range(6), [6, 5, 4, 3, 2, 1])

    for item in best_path:
        row = 6 - item["string"]
        col = item["fret"]
        label = midi_to_note_name(item["midi"])
        plt.text(col, row, label, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
