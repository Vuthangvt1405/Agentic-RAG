"""Stage loading effects: spinner + timing for pipeline stages.

Stdlib only. Automatically silent when output is not a TTY (piped logs,
tests) or when VERBOSE is False. ASCII-only so Windows consoles stay clean.
"""

import itertools
import sys
import threading
import time

VERBOSE = True
FRAMES = ["|", "/", "-", "\\"]


class stage:
    """Usage: with stage("Routing question"): ..."""

    def __init__(self, label):
        self.label = label
        self.enabled = False
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        frames = itertools.cycle(FRAMES)
        while not self._stop.is_set():
            sys.stdout.write(f"\r{next(frames)} {self.label}...")
            sys.stdout.flush()
            self._stop.wait(0.1)

    def __enter__(self):
        self.enabled = VERBOSE and sys.stdout.isatty()
        self.start = time.time()
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start
        if not self.enabled:
            return False
        self._stop.set()
        self._thread.join()
        status = "done" if exc_type is None else "failed"
        sys.stdout.write(f"\r[...] {self.label}... {status} ({elapsed:.1f}s)\n")
        sys.stdout.flush()
        return False
