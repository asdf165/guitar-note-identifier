import argparse
from collections import Counter

from src.audio_io import load_wav_mono, normalize_audio, bandpass_filter
from src.pitch import yin_pitch_track
from src.music import (
    frequency_to_midi,
    midi_to_note_name,
    smooth_midi_track,
    collapse_note_events,
)
from src.fretboard import (
    build_fretboard_map,
    map_note_to_positions,
    choose_best_position_path,
)
from src.visualize import (
    plot_waveform,
    plot_pitch_track,
    plot_fretboard_positions,
)


def main():
    parser = argparse.ArgumentParser(description="Guitar note to fretboard mapper")
    parser.add_argument("audio_path", type=str, help="Path to WAV file")
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=80.0)
    parser.add_argument("--fmax", type=float, default=1200.0)
    parser.add_argument("--topk", type=int, default=3, help="Top candidate fret positions per note")
    parser.add_argument("--no-filter", action="store_true", help="Disable band-pass filtering")
    args = parser.parse_args()

    sr, audio = load_wav_mono(args.audio_path)
    audio = normalize_audio(audio)

    if not args.no_filter:
        audio = bandpass_filter(audio, sr, lowcut=70.0, highcut=1300.0, order=4)

    times, freqs, confidences = yin_pitch_track(
        audio=audio,
        sr=sr,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        fmin=args.fmin,
        fmax=args.fmax,
        trough_threshold=0.12,
    )

    midi_track = frequency_to_midi(freqs)
    midi_track = smooth_midi_track(midi_track, window=5)

    note_names = [midi_to_note_name(m) if m is not None else None for m in midi_track]
    note_events = collapse_note_events(times, midi_track, min_duration=0.08)

    fretboard = build_fretboard_map(num_frets=20)

    print("\nDetected note events:")
    for ev in note_events:
        midi = ev["midi"]
        note = midi_to_note_name(midi)
        positions = map_note_to_positions(midi, fretboard)[:args.topk]
        print(
            f"{ev['start']:.2f}s - {ev['end']:.2f}s | {note:<4} | "
            f"candidates={positions}"
        )

    best_path = choose_best_position_path(note_events, fretboard)

    if best_path:
        print("\nMost likely playing path:")
        for item in best_path:
            note = midi_to_note_name(item["midi"])
            print(
                f"{item['start']:.2f}s - {item['end']:.2f}s | "
                f"{note:<4} -> string {item['string']} fret {item['fret']}"
            )

    pitch_counter = Counter([n for n in note_names if n is not None])
    print("\nMost common notes:")
    for note, count in pitch_counter.most_common(10):
        print(f"{note}: {count}")

    plot_waveform(audio, sr, title="Waveform")
    plot_pitch_track(times, freqs, confidences, title="Estimated Pitch Track")

    if best_path:
        plot_fretboard_positions(
            best_path,
            num_frets=20,
            title="Likely Fretboard Positions",
        )


if __name__ == "__main__":
    main()
