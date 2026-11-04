"""Token-free speaker diarization.

Approach: slide fixed-length windows over the speech regions, drop near-silent
windows, embed each with a pretrained ECAPA-TDNN speaker encoder (SpeechBrain,
not gated on HuggingFace), then cluster the embeddings.

The number of speakers is chosen automatically by trying several cluster counts
and keeping the one with the best silhouette separation. If even the best split
is weak, the audio is treated as a single speaker — this avoids splitting one
voice into several phantom speakers, which a fixed distance threshold is prone to.

Window labels are then smoothed (majority vote over neighbouring windows) and
attributed at *word* level when Whisper word timestamps are available: each word
gets the speaker of the windows centred around it, very short speaker turns are
absorbed by their neighbours, and segments are split where the speaker changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier

from app.models import Segment

logger = logging.getLogger(__name__)

_MIN_WINDOW_SECONDS = 0.5  # ECAPA needs a minimum amount of audio to be reliable
_EMBED_BATCH = 32
_MIN_RMS = 0.005  # windows quieter than this are treated as silence and skipped
_MIN_WINDOWS_FOR_MULTI = 4  # below this, don't attempt multi-speaker detection


@dataclass
class _Window:
    start: float
    end: float


class Diarizer:
    """Wraps a loaded ECAPA speaker encoder. Instantiate once and reuse."""

    def __init__(self, model_name: str, device: str, cache_dir: str | None = None) -> None:
        self.device = device
        self.encoder = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device},
            savedir=cache_dir,
        )

    def diarize(
        self,
        audio_path: Path,
        segments: list[Segment],
        *,
        window_seconds: float,
        hop_seconds: float,
        speaker_separation: float,
        num_speakers: int | None,
        min_speakers: int,
        max_speakers: int,
        smoothing_radius: int = 1,
        min_turn_seconds: float = 0.8,
    ) -> tuple[list[Segment], int]:
        """Attribute speakers to ``segments``.

        Segments with word timings are split where the speaker changes, so the
        returned list may be longer than the input one.

        Returns:
            The (possibly split) speaker-labelled segments and the number of
            distinct speakers found.
        """
        if not segments:
            return segments, 0

        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio = self._to_mono(waveform).squeeze(0)

        windows = self._build_windows(segments, window_seconds, hop_seconds)
        windows = self._drop_silent(audio, sample_rate, windows)
        if len(windows) <= 1:
            for seg in segments:
                seg.speaker = "SPEAKER 1"
            return segments, 1

        embeddings = self._embed_windows(audio, sample_rate, windows)
        labels = self._cluster(
            embeddings,
            speaker_separation=speaker_separation,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        order = sorted(range(len(windows)), key=lambda i: windows[i].start)
        windows = [windows[i] for i in order]
        labels = self._smooth(labels[order], smoothing_radius)
        return self._assign_speakers(segments, windows, labels, min_turn_seconds)

    # ---- Windowing -----------------------------------------------------
    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2 and waveform.size(0) > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform

    def _build_windows(
        self, segments: list[Segment], window_seconds: float, hop_seconds: float
    ) -> list[_Window]:
        """Create overlapping windows covering the transcribed speech regions."""
        windows: list[_Window] = []
        for seg in segments:
            span = seg.end - seg.start
            if span <= window_seconds:
                windows.append(_Window(seg.start, max(seg.end, seg.start + _MIN_WINDOW_SECONDS)))
                continue
            start = seg.start
            while start < seg.end:
                end = min(start + window_seconds, seg.end)
                if end - start >= _MIN_WINDOW_SECONDS:
                    windows.append(_Window(start, end))
                start += hop_seconds
        return windows

    def _drop_silent(
        self, audio: torch.Tensor, sample_rate: int, windows: list[_Window]
    ) -> list[_Window]:
        """Remove near-silent windows, whose embeddings are noise and cluster badly."""
        total = audio.size(0)
        kept: list[_Window] = []
        for win in windows:
            s = max(0, int(win.start * sample_rate))
            e = min(total, int(win.end * sample_rate))
            if e <= s:
                continue
            rms = float(torch.sqrt(torch.mean(audio[s:e] ** 2)))
            if rms >= _MIN_RMS:
                kept.append(win)
        # If everything looked silent (e.g. very quiet recording), keep the originals.
        return kept if kept else windows

    # ---- Embedding -----------------------------------------------------
    def _embed_windows(
        self, audio: torch.Tensor, sample_rate: int, windows: list[_Window]
    ) -> np.ndarray:
        total = audio.size(0)
        embeddings: list[np.ndarray] = []

        for i in range(0, len(windows), _EMBED_BATCH):
            batch = windows[i : i + _EMBED_BATCH]
            clips = []
            for win in batch:
                s = max(0, int(win.start * sample_rate))
                e = min(total, int(win.end * sample_rate))
                if e <= s:  # guard against zero-length slices at the file edge
                    e = min(total, s + 1)
                    s = max(0, e - 1)
                clips.append(audio[s:e])

            max_len = max(clip.size(0) for clip in clips)
            padded = torch.zeros(len(clips), max_len)
            lengths = torch.zeros(len(clips))
            for j, clip in enumerate(clips):
                padded[j, : clip.size(0)] = clip
                lengths[j] = clip.size(0) / max_len

            with torch.no_grad():
                out = self.encoder.encode_batch(padded.to(self.device), lengths.to(self.device))
            embeddings.append(out.squeeze(1).cpu().numpy())

        return np.vstack(embeddings)

    # ---- Clustering ----------------------------------------------------
    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 length-normalization (do NOT mean-center: it destroys the single-vs-
        multi separation signal and inflates within-speaker structure)."""
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    def _cluster(
        self,
        embeddings: np.ndarray,
        *,
        speaker_separation: float,
        num_speakers: int | None,
        min_speakers: int,
        max_speakers: int,
    ) -> np.ndarray:
        normalized = self._normalize(embeddings)
        n = len(normalized)

        # Explicit speaker count requested from the UI.
        if num_speakers and num_speakers >= 1:
            k = min(num_speakers, n)
            if k <= 1:
                return np.zeros(n, dtype=int)
            return self._agglomerative(normalized, k)

        # Too few windows to judge multiple speakers reliably -> assume one.
        if n < _MIN_WINDOWS_FOR_MULTI or max_speakers <= 1:
            return np.zeros(n, dtype=int)

        # Single-vs-multi gate: split into two coarse groups and measure how far
        # apart they are. One voice yields close centroids; distinct speakers are
        # far apart. This is far more reliable than a silhouette threshold.
        if min_speakers <= 1:
            two = self._agglomerative(normalized, 2)
            separation = self._separation(normalized, two)
            if separation < speaker_separation:
                logger.info("Diarization: single speaker (separation %.3f)", separation)
                return np.zeros(n, dtype=int)

        labels = self._select_k(normalized, min_speakers, max_speakers)
        logger.info("Diarization: %d speakers", len(set(labels)))
        return labels

    @staticmethod
    def _agglomerative(normalized: np.ndarray, k: int) -> np.ndarray:
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return model.fit_predict(normalized)

    @staticmethod
    def _separation(normalized: np.ndarray, labels: np.ndarray) -> float:
        """Cosine distance between the two cluster centroids."""
        c0 = normalized[labels == 0].mean(axis=0)
        c1 = normalized[labels == 1].mean(axis=0)
        cos = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-10))
        return 1.0 - cos

    def _select_k(
        self, normalized: np.ndarray, min_speakers: int, max_speakers: int
    ) -> np.ndarray:
        """Choose the speaker count by silhouette, preferring fewer speakers on ties."""
        n = len(normalized)
        lower = max(2, min_speakers)
        upper = min(max_speakers, n - 1)  # silhouette needs at most n-1 clusters

        scored: list[tuple[int, float, np.ndarray]] = []
        for k in range(lower, upper + 1):
            labels = self._agglomerative(normalized, k)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(normalized, labels, metric="cosine")
            scored.append((k, score, labels))

        if not scored:
            return self._agglomerative(normalized, min(lower, n))

        best_score = max(score for _, score, _ in scored)
        # Prefer the smallest k whose score is within a small margin of the best,
        # so we don't over-split when a coarser grouping is nearly as good.
        for k, score, labels in scored:
            if score >= best_score - 0.03:
                return labels
        return scored[0][2]

    # ---- Smoothing ----------------------------------------------------
    @staticmethod
    def _smooth(labels: np.ndarray, radius: int) -> np.ndarray:
        """Majority vote over ``radius`` neighbours on each side (time-ordered).

        Ties keep the window's own label, so genuine turns are preserved while
        isolated one-window flips (noisy embeddings) are removed.
        """
        if radius <= 0 or len(labels) < 3:
            return labels
        smoothed = labels.copy()
        for i in range(len(labels)):
            lo, hi = max(0, i - radius), min(len(labels), i + radius + 1)
            values, counts = np.unique(labels[lo:hi], return_counts=True)
            best = counts.max()
            winners = values[counts == best]
            if labels[i] not in winners:
                smoothed[i] = winners[0]
        return smoothed

    # ---- Attribution ---------------------------------------------------
    def _assign_speakers(
        self,
        segments: list[Segment],
        windows: list[_Window],
        labels: np.ndarray,
        min_turn_seconds: float,
    ) -> tuple[list[Segment], int]:
        """Attribute speakers (per word when possible) and give ordered names."""
        pieces: list[tuple[Segment, int]] = []
        for seg in segments:
            if seg.words:
                pieces.extend(self._split_by_words(seg, windows, labels, min_turn_seconds))
            else:
                pieces.append((seg, self._span_label(seg.start, seg.end, windows, labels)))

        # Rename raw cluster ids to "SPEAKER N" in order of first appearance.
        order: dict[int, str] = {}
        for _, raw in pieces:
            if raw not in order:
                order[raw] = f"SPEAKER {len(order) + 1}"

        result: list[Segment] = []
        for seg, raw in pieces:
            seg.speaker = order[raw]
            result.append(seg)
        return result, len(order)

    def _split_by_words(
        self,
        seg: Segment,
        windows: list[_Window],
        labels: np.ndarray,
        min_turn_seconds: float,
    ) -> list[tuple[Segment, int]]:
        """Label each word, absorb very short turns and split at speaker changes."""
        word_labels = [self._span_label(w.start, w.end, windows, labels) for w in seg.words]

        # Group consecutive words with the same label into runs.
        runs: list[list] = []  # [label, [words]]
        for word, label in zip(seg.words, word_labels, strict=True):
            if runs and runs[-1][0] == label:
                runs[-1][1].append(word)
            else:
                runs.append([label, [word]])

        # Absorb runs shorter than ``min_turn_seconds`` into the longer neighbour,
        # shortest first, until every run is long enough (or only one is left).
        def run_span(run: list) -> float:
            return run[1][-1].end - run[1][0].start

        while len(runs) > 1:
            shortest = min(range(len(runs)), key=lambda i: run_span(runs[i]))
            if run_span(runs[shortest]) >= min_turn_seconds:
                break
            neighbours = [i for i in (shortest - 1, shortest + 1) if 0 <= i < len(runs)]
            target = max(neighbours, key=lambda i: run_span(runs[i]))
            if target < shortest:
                runs[target][1].extend(runs[shortest][1])
            else:
                runs[target][1][:0] = runs[shortest][1]
            del runs[shortest]
            # Merge neighbours that now share the same label.
            merged: list[list] = []
            for run in runs:
                if merged and merged[-1][0] == run[0]:
                    merged[-1][1].extend(run[1])
                else:
                    merged.append(run)
            runs = merged

        if len(runs) == 1:
            return [(seg, runs[0][0])]

        pieces: list[tuple[Segment, int]] = []
        for i, (label, words) in enumerate(runs):
            start = seg.start if i == 0 else words[0].start
            end = seg.end if i == len(runs) - 1 else words[-1].end
            text = "".join(w.text for w in words).strip()
            if not text:
                continue
            piece = Segment(start=start, end=end, text=text, words=list(words))
            pieces.append((piece, label))
        return pieces or [(seg, runs[0][0])]

    @staticmethod
    def _span_label(
        start: float, end: float, windows: list[_Window], labels: np.ndarray
    ) -> int:
        """Pick the cluster that best covers ``[start, end]``.

        Overlap is weighted by how close each window's centre is to the span's
        centre, so the windows centred on a word dominate those that only graze it.
        """
        mid = (start + end) / 2
        scores: dict[int, float] = {}
        for win, label in zip(windows, labels, strict=True):
            span = min(end, win.end) - max(start, win.start)
            if span <= 0:
                continue
            half = (win.end - win.start) / 2 or 1e-6
            centre = (win.start + win.end) / 2
            weight = max(0.1, 1.0 - abs(centre - mid) / half)
            scores[int(label)] = scores.get(int(label), 0.0) + span * weight

        if scores:
            return max(scores, key=scores.get)

        # No overlap (short span between windows): use the nearest window.
        nearest = min(
            range(len(windows)),
            key=lambda i: abs(mid - (windows[i].start + windows[i].end) / 2),
        )
        return int(labels[nearest])
