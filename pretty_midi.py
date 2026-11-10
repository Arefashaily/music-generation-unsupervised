"""Minimal pretty_midi-compatible shim using mido.

This provides just enough API surface for the project code to run on
Python versions where the real pretty_midi package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mido
import numpy as np


@dataclass
class Note:
    velocity: int
    pitch: int
    start: float
    end: float


class Instrument:
    def __init__(self, program: int = 0, is_drum: bool = False):
        self.program = program
        self.is_drum = is_drum
        self.notes: List[Note] = []


class PrettyMIDI:
    def __init__(self, midi_file: Optional[str] = None, initial_tempo: float = 120.0):
        self.instruments: List[Instrument] = []
        self.time_signature_changes: List[object] = []
        self.key_signature_changes: List[object] = []
        self._tempo_bpm = float(initial_tempo)

        if midi_file is not None:
            self._load_mido(midi_file)

    def _load_mido(self, path: str) -> None:
        mid = mido.MidiFile(path)
        tempo = mido.bpm2tempo(self._tempo_bpm)

        # Use a single instrument for simplicity (piano-like).
        instrument = Instrument(program=0, is_drum=False)
        notes_on = {}

        for track in mid.tracks:
            abs_time = 0.0
            for msg in track:
                abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    self._tempo_bpm = float(mido.tempo2bpm(tempo))
                elif msg.type == "note_on" and msg.velocity > 0:
                    notes_on[msg.note] = (abs_time, msg.velocity)
                elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                    if msg.note in notes_on:
                        start, vel = notes_on.pop(msg.note)
                        instrument.notes.append(
                            Note(
                                velocity=int(vel),
                                pitch=int(msg.note),
                                start=float(start),
                                end=float(abs_time),
                            )
                        )

        if instrument.notes:
            self.instruments.append(instrument)

    def get_piano_roll(self, fs: int = 100) -> np.ndarray:
        end_time = self.get_end_time()
        n_frames = max(1, int(np.ceil(end_time * fs)))
        roll = np.zeros((128, n_frames), dtype=np.float32)

        for inst in self.instruments:
            for note in inst.notes:
                start_idx = int(np.floor(note.start * fs))
                end_idx = int(np.ceil(note.end * fs))
                end_idx = max(end_idx, start_idx + 1)
                roll[note.pitch, start_idx:end_idx] = max(roll[note.pitch, start_idx:end_idx].max(), note.velocity)

        return roll

    def estimate_tempo(self) -> float:
        return float(self._tempo_bpm)

    def get_end_time(self) -> float:
        max_end = 0.0
        for inst in self.instruments:
            for note in inst.notes:
                max_end = max(max_end, note.end)
        return max_end

    def write(self, path: str) -> None:
        ticks_per_beat = 480
        tempo = mido.bpm2tempo(self._tempo_bpm)

        events = []
        for inst in self.instruments:
            for note in inst.notes:
                events.append((note.start, True, note.pitch, note.velocity))
                events.append((note.end, False, note.pitch, 0))

        events.sort(key=lambda x: (x[0], not x[1]))

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        last_time = 0.0
        for t, is_on, pitch, velocity in events:
            delta = t - last_time
            ticks = int(mido.second2tick(delta, ticks_per_beat, tempo))
            msg_type = "note_on" if is_on else "note_off"
            track.append(mido.Message(msg_type, note=int(pitch), velocity=int(velocity), time=ticks))
            last_time = t

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        mid.tracks.append(track)
        mid.save(path)
