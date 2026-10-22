import logging

from src.constants import LOGGER_NAME
from src.utils import SafeStreamHandler, setup_logging


def test_setup_logging_writes_file_and_uses_named_logger(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    log = setup_logging("INFO", str(log_file))
    log.info("hello from the test")
    for handler in log.handlers:
        handler.flush()

    assert log.name == LOGGER_NAME
    assert log.propagate is False
    assert logging.getLogger().handlers == []
    assert "INFO     - hello from the test" in log_file.read_text(encoding="utf-8")


def test_setup_logging_does_not_duplicate_handlers(tmp_path):
    setup_logging("INFO", str(tmp_path / "a.log"))
    log = setup_logging("DEBUG", str(tmp_path / "b.log"))
    assert len(log.handlers) == 2
    assert sum(isinstance(h, SafeStreamHandler) for h in log.handlers) == 1
    assert log.level == logging.DEBUG
