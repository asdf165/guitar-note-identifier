import math
from collections import Counter

from .config import NOTE_NAMES_SHARP


def frequency_to_midi(freqs):
    midi_values = []
    for f in freqs:
        if f is None or f <= 0:
            midi_values.append(None)
            continue
        midi = int(round(69 + 12 * math.log2(f / 440.0)))
        midi_values.append(midi)
    return midi_values


def midi_to_note_name(midi: int | None):
    if midi is None:
        return None
    octave = (midi // 12) - 1
    name = NOTE_NAMES_SHARP[midi % 12]
    return f"{name}{octave}"


def smooth_midi_track(midi_track, window=5):
    if window < 1:
        return midi_track[:]

    half = window // 2
    out = []

    for i in range(len(midi_track)):
        neighbors = [
            midi_track[j]
            for j in range(max(0, i - half), min(len(midi_track), i + half + 1))
            if midi_track[j] is not None
        ]
        if not neighbors:
            out.append(None)
        else:
            out.append(Counter(neighbors).most_common(1)[0][0])

    return out


def collapse_note_events(times, midi_track, min_duration=0.08):
    events = []
    if len(times) == 0:
        return events

    start_idx = 0
    current = midi_track[0]

    for i in range(1, len(midi_track)):
        if midi_track[i] != current:
            start_time = times[start_idx]
            end_time = times[i - 1]
            duration = end_time - start_time

            if current is not None and duration >= min_duration:
                events.append({
                    "start": start_time,
                    "end": end_time,
                    "midi": current,
                })

            start_idx = i
            current = midi_track[i]

    start_time = times[start_idx]
    end_time = times[-1]
    duration = end_time - start_time
    if current is not None and duration >= min_duration:
        events.append({
            "start": start_time,
            "end": end_time,
            "midi": current,
        })

    return events
