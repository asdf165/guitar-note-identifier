from .config import STANDARD_TUNING_MIDI


def build_fretboard_map(num_frets=20):
    """
    Returns:
    {
        string_number: {fret_number: midi_note}
    }
    """
    fretboard = {}
    for string_num, open_midi in STANDARD_TUNING_MIDI.items():
        fretboard[string_num] = {}
        for fret in range(num_frets + 1):
            fretboard[string_num][fret] = open_midi + fret
    return fretboard


def map_note_to_positions(midi_note: int, fretboard: dict):
    positions = []
    for string_num, frets in fretboard.items():
        for fret, midi in frets.items():
            if midi == midi_note:
                positions.append((string_num, fret))
    positions.sort(key=lambda x: (x[1], x[0]))
    return positions


def choose_best_position_path(note_events, fretboard):
    """
    Greedy position chooser:
    - prefer lower fret
    - prefer smaller movement from previous note
    - prefer nearby string
    """
    if not note_events:
        return []

    chosen = []
    prev = None

    for ev in note_events:
        candidates = map_note_to_positions(ev["midi"], fretboard)
        if not candidates:
            continue

        best = None
        best_cost = float("inf")

        for string_num, fret in candidates:
            if prev is None:
                cost = fret
            else:
                cost = (
                    abs(fret - prev["fret"]) * 2.0 +
                    abs(string_num - prev["string"]) * 1.0 +
                    fret * 0.15
                )

            if cost < best_cost:
                best_cost = cost
                best = (string_num, fret)

        chosen_item = {
            "start": ev["start"],
            "end": ev["end"],
            "midi": ev["midi"],
            "string": best[0],
            "fret": best[1],
        }
        chosen.append(chosen_item)
        prev = chosen_item

    return chosen
