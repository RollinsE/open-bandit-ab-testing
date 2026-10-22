import logging

from src.utils import SafeStreamHandler


class BrokenStream:
    def write(self, _message):
        raise OSError("transport endpoint is not connected")

    def flush(self):
        raise OSError("transport endpoint is not connected")


def test_safe_stream_handler_swallows_broken_colab_stream():
    logger = logging.getLogger("test_safe_stream_handler")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = SafeStreamHandler(BrokenStream())
    logger.addHandler(handler)

    logger.info("this should not raise or print a logging traceback")
    logger.info("subsequent messages should also be ignored cleanly")

    assert handler._stream_broken is True
