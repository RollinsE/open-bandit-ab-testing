import logging
import time

from joblib import delayed

from src.utils import Heartbeat, Progress, format_duration, iter_parallel


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _logger():
    log = logging.getLogger("test_progress")
    log.handlers.clear()
    log.propagate = False
    log.setLevel(logging.INFO)
    handler = _ListHandler()
    log.addHandler(handler)
    return log, handler


def _sleep_and_return(value, seconds):
    time.sleep(seconds)
    return value


def test_progress_logs_each_task_with_eta():
    log, handler = _logger()
    progress = Progress("Step", 4, log)
    progress.update("a")
    progress.update("b")
    assert handler.messages[0].startswith("Step: 1 of 4 done (a)")
    assert "left" in handler.messages[1]


def test_heartbeat_speaks_when_nothing_finishes():
    log, handler = _logger()
    progress = Progress("Slow step", 1, log)
    with Heartbeat(progress, interval=0.2):
        time.sleep(0.75)
    still = [m for m in handler.messages if "still running" in m]
    assert len(still) >= 2
    assert still[0].startswith("Slow step: still running, 0 of 1 done")


def test_iter_parallel_yields_every_result_as_it_finishes():
    tasks = [delayed(_sleep_and_return)(i, 0.3 if i == 0 else 0.0) for i in range(4)]
    assert sorted(iter_parallel(2, tasks)) == [0, 1, 2, 3]
    assert list(iter_parallel(1, tasks)) == [0, 1, 2, 3]


def test_format_duration():
    assert format_duration(42) == "42s"
    assert format_duration(725) == "12m05s"
    assert format_duration(7380) == "2h03m"

