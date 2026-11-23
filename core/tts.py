"""Local text-to-speech via kokoro-onnx (ONNX Runtime, CPU by default) so
speech synthesis never competes with the Ollama models for VRAM.

Streaming pipeline: text is fed in as it arrives (e.g. LLM tokens), split
into sentences, synthesized one sentence at a time on a synthesis thread,
and played on a separate playback thread — so the first sentence starts
playing while later ones are still being generated/synthesized, instead
of waiting for the whole reply. (Before this, a chat reply waited for the
full LLM response *and* full synthesis before any audio; with the old
int8 model that was routinely 20s+ of silence.)

- start_utterance() -> Utterance; .feed(text) any number of times, then
  .close(). Utterances play in the order they were started.
- speak(text): one-shot helper; blocks until it's been spoken (scheduler
  jobs rely on that to read reminders one after another).
- stop(): cancels everything queued or playing, within ~0.1s. The chat UI
  calls it when a new message is sent so replies never pile up.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

import config

logger = logging.getLogger(__name__)

# Sentence end: terminal punctuation (plus any closing quote/bracket)
# followed by whitespace, or a line break. A '.' inside "41.5" has no
# whitespace after it, so decimals don't split.
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
# The first chunk of an utterance may also end at a clause break, once
# it's long enough to sound natural — audio can't start until the first
# chunk is fully synthesized, so a shorter first chunk means sound sooner
# (~1.5s for a typical first sentence). Later chunks stay whole sentences.
_CLAUSE_END = re.compile(r"[,;:]\s+")
_FIRST_CHUNK_MIN_CHARS = 30
_MARKDOWN = re.compile(r"[*_`#>|]+")
_PLAYBACK_SLICE_SECONDS = 0.1  # stop() latency: one slice at most
_INTER_SENTENCE_PAUSE_SECONDS = 0.2
_IDLE_CLOSE_SECONDS = 1.0  # keep the audio stream open across short gaps

# Voice levels for the UI's dot grid: each playback slice is cut into 25ms
# windows; each window gets a loudness and 8 log-spaced band levels, all
# 0..1. Per-band dB ranges were measured on af_heart output (low end =
# median - 6 dB, high end = 99th percentile), so every band uses the full
# range even though speech has far less energy up high. Costs ~0.4ms per
# 0.1s slice. Re-measure if KOKORO_VOICE changes a lot.
VOICE_BANDS = 8
_LEVEL_WINDOW_SECONDS = 0.025
_BAND_EDGES_HZ = np.geomspace(80, 8000, VOICE_BANDS + 1)
_BAND_DB_LO = np.array([-65, -46, -53, -63, -72, -75, -78, -88], dtype=np.float32)
_BAND_DB_HI = np.array([-42, -23, -32, -35, -35, -46, -50, -49], dtype=np.float32)
_LOUD_DB_LO, _LOUD_DB_HI = -45.0, -15.0
_level_tables: dict = {}


def _voice_levels(audio: np.ndarray, sample_rate: int) -> list[tuple[float, float, np.ndarray]]:
    """[(offset seconds into audio, loudness, bands[8]), ...] per 25ms window."""
    win = int(sample_rate * _LEVEL_WINDOW_SECONDS)
    if win < 16 or len(audio) < win:
        return []
    tables = _level_tables.get((sample_rate, win))
    if tables is None:
        freqs = np.fft.rfftfreq(win, 1 / sample_rate)
        bins = [
            np.flatnonzero((freqs >= _BAND_EDGES_HZ[i]) & (freqs < _BAND_EDGES_HZ[i + 1]))
            for i in range(VOICE_BANDS)
        ]
        tables = _level_tables[(sample_rate, win)] = (np.hanning(win).astype(np.float32), bins)
    hann, bins = tables
    out = []
    for start in range(0, len(audio) - win + 1, win):
        w = audio[start : start + win]
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(w * w))) + 1e-9)
        mag = np.abs(np.fft.rfft(w * hann)) / (win / 2)
        db = np.array(
            [20 * np.log10(float(mag[b].mean()) + 1e-9) if len(b) else -120.0 for b in bins],
            dtype=np.float32,
        )
        bands = np.clip((db - _BAND_DB_LO) / (_BAND_DB_HI - _BAND_DB_LO), 0.0, 1.0)
        loud = min(1.0, max(0.0, (rms_db - _LOUD_DB_LO) / (_LOUD_DB_HI - _LOUD_DB_LO)))
        out.append((start / sample_rate, loud, bands))
    return out


def _clean_for_speech(text: str) -> str:
    return " ".join(_MARKDOWN.sub(" ", text).split())


class Utterance:
    """One spoken message, fed incrementally. Not thread-safe on its own:
    feed/close it from a single thread (the one producing the text)."""

    def __init__(self, engine: "TTSEngine", generation: int):
        self._engine = engine
        self._generation = generation
        self._buffer = ""
        self._sent_any = False  # first chunk may split at a clause break
        self._pending = 0  # sentences queued but not finished playing
        self._closed = False
        self._active = False  # counted in engine's "speaking" state
        self.done = threading.Event()

    def feed(self, text: str) -> None:
        if self._closed or not text:
            return
        self._buffer += text
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not self._sent_any:
                clause = _CLAUSE_END.search(self._buffer, _FIRST_CHUNK_MIN_CHARS)
                if clause and (not match or clause.start() < match.start()):
                    match = clause
            if not match:
                break
            sentence, self._buffer = self._buffer[: match.end()], self._buffer[match.end() :]
            self._sent_any = True
            self._engine._enqueue_sentence(self, sentence)
        if len(self._buffer) > config.TTS_MAX_CHUNK_CHARS:
            # No sentence end in a long run (a list, code, rambling) —
            # cut at the last comma/space so speech isn't held back.
            cut = max(self._buffer.rfind(",", 0, config.TTS_MAX_CHUNK_CHARS),
                      self._buffer.rfind(" ", 0, config.TTS_MAX_CHUNK_CHARS))
            cut = cut + 1 if cut > 0 else config.TTS_MAX_CHUNK_CHARS
            self._engine._enqueue_sentence(self, self._buffer[:cut])
            self._buffer = self._buffer[cut:]

    def close(self) -> None:
        if self._closed:
            return
        if self._buffer.strip():
            self._engine._enqueue_sentence(self, self._buffer)
        self._buffer = ""
        self._closed = True
        self._engine._maybe_finish(self)


class TTSEngine:
    def __init__(self):
        self._state_lock = threading.Lock()
        self._kokoro = None
        self._muted = False
        self._listeners: list[Callable[[bool], None]] = []
        self._level_listeners: list[Callable[[list], None]] = []
        self._generation = 0  # bumped by stop(); stale work is dropped
        self._active_utterances = 0
        self._synth_queue: queue.Queue = queue.Queue()
        self._play_queue: queue.Queue = queue.Queue()
        try:
            self._kokoro = Kokoro(
                str(config.KOKORO_MODEL_PATH), str(config.KOKORO_VOICES_PATH)
            )
        except Exception:
            logger.exception(
                "kokoro-onnx failed to load (missing model files in %s?); "
                "TTS is disabled",
                config.MODELS_DIR,
            )
            return
        threading.Thread(target=self._synth_loop, name="tts-synth", daemon=True).start()
        threading.Thread(target=self._play_loop, name="tts-play", daemon=True).start()

    @property
    def available(self) -> bool:
        return self._kokoro is not None

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Muting also cuts off whatever is playing right now. Applies to
        every caller (chat replies, scheduled briefing, reminder alerts) —
        in-memory only, resets on restart."""
        self._muted = value
        if value:
            self.stop()

    def add_listener(self, fn: Callable[[bool], None]) -> None:
        """fn(True) when speech starts (from the first queued sentence,
        including its synthesis time), fn(False) when everything queued
        has finished or been stopped. Called on a TTS thread — UI
        listeners must hand off to their own main thread, not touch
        widgets directly. Register at startup only (not thread-safe)."""
        self._listeners.append(fn)

    def add_level_listener(self, fn: Callable[[list], None]) -> None:
        """fn(frames) as audio is handed to the sound device, where frames
        is [(time.perf_counter() when audible, loudness 0..1, bands[8]
        0..1), ...] — one per 25ms of audio. Called on the playback thread,
        same rules as add_listener. Register at startup only."""
        self._level_listeners.append(fn)

    def _notify_levels(self, stream, audio: np.ndarray, sample_rate: int) -> None:
        if not self._level_listeners:
            return
        frames = _voice_levels(audio, sample_rate)
        if not frames:
            return
        # write() just returned, so this slice sits at the back of the
        # device buffer: it becomes audible after roughly the output latency.
        try:
            latency = float(stream.latency)
        except Exception:
            latency = 0.1
        audible_at = time.perf_counter() + latency
        timed = [(audible_at + offset, loud, bands) for offset, loud, bands in frames]
        for fn in self._level_listeners:
            try:
                fn(timed)
            except Exception:
                logger.exception("TTS level listener failed")

    def _notify(self, speaking: bool) -> None:
        for fn in self._listeners:
            try:
                fn(speaking)
            except Exception:
                logger.exception("TTS listener failed")

    # -- public API ------------------------------------------------------------

    def start_utterance(self) -> Utterance:
        return Utterance(self, self._generation)

    def speak(self, text: str) -> None:
        """Speak text and block until it has finished (or been stopped).
        Safe to call even if the model failed to load — a no-op rather
        than crashing the caller (scheduler jobs, UI worker threads)."""
        if not text or not self.available or self._muted:
            return
        utterance = self.start_utterance()
        utterance.feed(text)
        utterance.close()
        utterance.done.wait()

    def stop(self) -> None:
        """Cancel everything queued or playing. Utterances started before
        this call go silent; ones started after it play normally."""
        with self._state_lock:
            self._generation += 1
            was_speaking = self._active_utterances > 0
            self._active_utterances = 0
        if was_speaking:
            self._notify(False)

    # -- internals -------------------------------------------------------------

    def _is_current(self, utterance: Utterance) -> bool:
        return utterance._generation == self._generation

    def _enqueue_sentence(self, utterance: Utterance, sentence: str) -> None:
        text = _clean_for_speech(sentence)
        if not text or not any(ch.isalnum() for ch in text):
            return
        if not self.available or self._muted:
            return
        became_speaking = False
        with self._state_lock:
            if not self._is_current(utterance):
                return
            utterance._pending += 1
            if not utterance._active:
                utterance._active = True
                self._active_utterances += 1
                became_speaking = self._active_utterances == 1
        if became_speaking:
            self._notify(True)
        self._synth_queue.put((utterance, text))

    def _sentence_finished(self, utterance: Utterance) -> None:
        with self._state_lock:
            utterance._pending -= 1
        self._maybe_finish(utterance)

    def _maybe_finish(self, utterance: Utterance) -> None:
        """Mark the utterance done once it's closed and fully played (or
        was cancelled by stop())."""
        became_idle = False
        with self._state_lock:
            stale = not self._is_current(utterance)
            if not (stale or (utterance._closed and utterance._pending <= 0)):
                return
            if utterance._active and not stale:
                self._active_utterances -= 1
                became_idle = self._active_utterances == 0
            utterance._active = False
        utterance.done.set()
        if became_idle:
            self._notify(False)

    def _synth_loop(self) -> None:
        pause = None
        while True:
            utterance, text = self._synth_queue.get()
            if not self._is_current(utterance) or self._muted:
                self._maybe_finish(utterance)
                continue
            try:
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=config.KOKORO_VOICE,
                    speed=config.KOKORO_SPEED,
                    lang=config.KOKORO_LANG,
                )
            except Exception:
                logger.exception("TTS synthesis failed for %r", text[:80])
                self._sentence_finished(utterance)
                continue
            if pause is None or len(pause) != int(sample_rate * _INTER_SENTENCE_PAUSE_SECONDS):
                pause = np.zeros(int(sample_rate * _INTER_SENTENCE_PAUSE_SECONDS), dtype=np.float32)
            audio = np.concatenate([samples.astype(np.float32, copy=False), pause])
            self._play_queue.put((utterance, audio, sample_rate))

    def _play_loop(self) -> None:
        while True:
            item = self._play_queue.get()  # block while idle (no stream open)
            sample_rate = item[2]
            try:
                with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                    while item is not None:
                        utterance, audio, rate = item
                        if rate != sample_rate:
                            # Never expected (Kokoro is fixed-rate); requeue
                            # so it plays on a stream opened at its rate.
                            self._play_queue.put(item)
                            item = None
                            break
                        # _play_one always finishes the sentence itself.
                        item = None
                        self._play_one(stream, utterance, audio, sample_rate)
                        try:
                            item = self._play_queue.get(timeout=_IDLE_CLOSE_SECONDS)
                        except queue.Empty:
                            item = None
            except Exception:
                logger.exception("TTS playback failed")
                if item is not None:  # taken off the queue but never played
                    self._sentence_finished(item[0])

    def _play_one(self, stream, utterance: Utterance, audio: np.ndarray, sample_rate: int) -> None:
        step = int(sample_rate * _PLAYBACK_SLICE_SECONDS)
        try:
            for start in range(0, len(audio), step):
                if not self._is_current(utterance) or self._muted:
                    if start:  # something of this sentence is already buffered
                        stream.abort()  # drop it instead of letting it finish
                        stream.start()
                    break
                chunk = audio[start : start + step]
                stream.write(chunk)
                self._notify_levels(stream, chunk, sample_rate)
        finally:
            self._sentence_finished(utterance)
