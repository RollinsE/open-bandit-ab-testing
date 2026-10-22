# -*- coding: utf-8 -*-
"""Shared utility functions for logging, timestamps and file output.

Logging follows the same structure as the Brownsea pipeline:

1. File first. Every run writes a full log to ``builds/<run_id>/logs/``, so the
   record survives even if the notebook output is lost.
2. Colab safe. The console handler is a ``SafeStreamHandler`` that stops writing
   quietly when Colab detaches stdout, instead of printing a traceback for
   every message.
3. One named logger. All modules log through ``LOGGER_NAME`` and never through
   the root logger, so third-party libraries cannot add or duplicate handlers.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.constants import LOGGER_NAME


# =============================================================================
# LOGGING
# =============================================================================

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that does not dump traceback noise when Colab disconnects stdout/stderr.

    Google Colab occasionally detaches the notebook output stream while a long-running
    subprocess is still alive. The standard logging StreamHandler reports that as a
    large "--- Logging error ---" traceback for every log message. The pipeline should
    keep running and continue writing the file log instead.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._stream_broken = False

    def emit(self, record):
        if self._stream_broken:
            return
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stream_broken = True
        except Exception:
            # Do not allow console logging failures to interrupt or pollute pipeline runs.
            self._stream_broken = True

    def flush(self):
        if self._stream_broken:
            return
        try:
            super().flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stream_broken = True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure robust file-first logging for the pipeline.

    Args:
        log_level: One of DEBUG, INFO, WARNING or ERROR.
        log_file: Path of the run's log file. Its folder is created if needed.
            When ``None`` only the console handler is attached.

    Returns:
        The pipeline logger, named ``LOGGER_NAME``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Avoid logging.basicConfig(force=True), which installs a standard stream handler
    # that can explode with repeated tracebacks in Colab subprocess output.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)

    LOG = logging.getLogger(LOGGER_NAME)
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    LOG.setLevel(level)
    LOG.propagate = False

    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    LOG.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        LOG.addHandler(file_handler)

    return LOG


def get_logger() -> logging.Logger:
    """Return the pipeline logger for use inside modules."""
    return logging.getLogger(LOGGER_NAME)


# =============================================================================
# PROGRESS
# =============================================================================

def format_duration(seconds: float) -> str:
    """Human-readable duration: 45s, 12m05s, 2h03m."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Progress:
    """Counts finished tasks for one long step and logs each one with elapsed time and an ETA.

    Used together with ``iter_parallel`` (one ``update`` per finished task) and
    ``Heartbeat`` (a "still running" line when nothing has finished for a while).
    The ETA assumes the remaining tasks take as long, on average, as the finished
    ones; it is printed only for steps whose tasks are similar in size.
    """

    def __init__(self, label: str, total: int, log: Optional[logging.Logger] = None, eta: bool = True,
                 already_done: int = 0):
        self.label, self.total, self.done = label, total, already_done
        self.base = already_done  # finished in an earlier session: not counted in the time estimate
        self.eta = eta
        self.log = log or get_logger()
        self.start = time.monotonic()
        self.last_update = self.start

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def update(self, what: str = "", n: int = 1) -> None:
        self.done += n
        self.last_update = time.monotonic()
        message = f"{self.label}: {self.done} of {self.total} done"
        if what:
            message += f" ({what})"
        message += f", {format_duration(self.elapsed)} elapsed"
        if self.eta and self.base < self.done < self.total:
            remaining = self.elapsed / (self.done - self.base) * (self.total - self.done)
            message += f", about {format_duration(remaining)} left"
        self.log.info(message)

    def status(self) -> str:
        return (f"{self.label}: still running, {self.done} of {self.total} done, "
                f"{format_duration(self.elapsed)} elapsed")


class Heartbeat:
    """Context manager that logs a "still running" line every ``interval`` seconds.

    Long steps run their work in worker processes, which do not write to the log,
    and a single task (tuning one model family, say) can take many minutes. The
    heartbeat runs in a thread of the main process, so the log never goes quiet
    for longer than ``interval`` seconds and a stalled run is easy to tell apart
    from a slow one.
    """

    def __init__(self, progress: Progress, interval: float = 180.0):
        self.progress, self.interval = progress, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if time.monotonic() - self.progress.last_update >= self.interval * 0.9:
                self.progress.log.info(self.progress.status())

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def iter_parallel(n_jobs: int, tasks: list):
    """Run joblib ``delayed`` tasks in parallel, yielding each result as soon as it finishes.

    Falls back to in-order results on older joblib versions, and to plain
    sequential execution when ``n_jobs`` is 1.
    """
    from joblib import Parallel

    if n_jobs == 1:
        for func, args, kwargs in tasks:
            yield func(*args, **kwargs)
        return
    for mode in ("generator_unordered", "generator"):
        try:
            runner = Parallel(n_jobs=n_jobs, return_as=mode)
        except (ValueError, TypeError):  # older joblib: unknown or unsupported return_as
            continue
        yield from runner(tasks)
        return
    yield from Parallel(n_jobs=n_jobs)(tasks)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp() -> str:
    """Get formatted timestamp for file naming and run ids."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def available_memory_bytes() -> Optional[int]:
    """Memory the operating system reports as available, or ``None`` if it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def memory_safe_n_jobs(requested: int, bytes_per_job: float, headroom: float = 0.6) -> int:
    """Cap parallel jobs so their combined working memory fits in available RAM.

    Args:
        requested: Jobs asked for; -1 means one per CPU core.
        bytes_per_job: Estimated peak memory of one job.
        headroom: Share of available memory the jobs may use together.

    Returns:
        At least 1, at most the requested number of jobs.
    """
    cores = os.cpu_count() or 1
    wanted = cores if requested is None or requested < 0 else max(1, requested)
    available = available_memory_bytes()
    if not available or bytes_per_job <= 0:
        return wanted
    return int(max(1, min(wanted, (available * headroom) // bytes_per_job)))


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON, creating parent folders.

    Args:
        path: Destination file.
        payload: Any JSON-serialisable object. NumPy scalars are converted.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    """Convert NumPy and Path values that the json module cannot serialise."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
