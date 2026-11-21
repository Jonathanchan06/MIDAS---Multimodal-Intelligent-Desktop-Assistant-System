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
                stream.write(audio[start : start + step])
        finally:
            self._sentence_finished(utterance)
