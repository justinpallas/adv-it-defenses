"""Lightweight terminal progress indicator with ETA support."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import TextIO


def _format_duration(seconds: float) -> str:
    if math.isinf(seconds) or seconds >= 1000 * 3600:
        return "--:--"
    seconds = max(0, int(seconds))
    minutes, rem = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{rem:02d}"
    return f"{minutes:02d}:{rem:02d}"


@dataclass
class ProgressState:
    description: str
    unit: str
    total: int
    completed: int = 0


class Progress:
    """Render a simple progress bar to a terminal."""

    def __init__(
        self,
        *,
        total: int,
        description: str,
        unit: str = "items",
        stream: TextIO | None = None,
        min_interval: float = 0.1,
        bar_width: int = 24,
    ) -> None:
        if total < 0:
            raise ValueError("Progress total must be non-negative.")
        self.state = ProgressState(description=description, unit=unit, total=total)
        self._stream = stream or sys.stderr
        self._min_interval = max(0.01, min_interval)
        self._bar_width = max(10, bar_width)
        self._start_time = time.monotonic()
        self._last_render = 0.0
        self._last_line_len = 0
        self._closed = False
        self._eta_override: float | None = None
        if total == 0:
            self.state.total = 1
        self._render(force=True)

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def update(self, advance: int = 1) -> None:
        if self._closed:
            return
        if advance < 0:
            raise ValueError("Progress advance must be non-negative.")
        self.state.completed = min(
            self.state.total,
            self.state.completed + advance,
        )
        self._render()

    def add_total(self, delta: int) -> None:
        if self._closed:
            raise RuntimeError("Cannot extend a closed progress bar.")
        if delta < 0:
            raise ValueError("Progress total increment must be non-negative.")
        if delta == 0:
            return
        self.state.total += delta
        self._render(force=True)

    def complete(self) -> None:
        if self._closed:
            return
        self.state.completed = self.state.total
        self._render(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._render(force=True)
        self._stream.write("\n")
        self._stream.flush()
        self._last_line_len = 0
        self._closed = True

    def refresh(self) -> None:
        """Force a redraw of the progress bar."""
        if self._closed:
            return
        self._render(force=True)

    def set_eta_override(self, eta_seconds: float | None) -> None:
        """Override the displayed ETA (provide seconds, or None to reset)."""
        if eta_seconds is None or math.isinf(eta_seconds):
            self._eta_override = None
        else:
            self._eta_override = max(0.0, eta_seconds)

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_render) < self._min_interval:
            return

        completed = self.state.completed
        total = max(1, self.state.total)
        fraction = completed / total
        percent = fraction * 100.0
        elapsed = now - self._start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - completed)
        eta = remaining / rate if rate > 0 else math.inf
        if self._eta_override is not None:
            eta = self._eta_override

        filled = int(self._bar_width * fraction)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        eta_str = _format_duration(eta)
        elapsed_str = _format_duration(elapsed)
        message = (
            f"\r{self.state.description:<24} "
            f"{percent:6.2f}% [{bar}] "
            f"{completed}/{self.state.total} {self.state.unit}"
            f" | elapsed {elapsed_str} | eta {eta_str}"
        )
        padding = max(0, self._last_line_len - len(message))
        self._stream.write(message + " " * padding)
        self._stream.flush()
        self._last_line_len = len(message)
        self._last_render = now
