import logging
import os
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

def setup_logging(stream=None) -> None:
    """Configure app logging once. Level comes from LOG_LEVEL (default INFO).

    INFO logs what happened (intents, bookings saved, connections) without customer details;
    DEBUG adds messages, transcripts and extracted booking data.
    """
    stream = stream or sys.stderr
    # Replies can contain emoji. On Windows, a stream redirected to a file uses a legacy
    # encoding that can't represent them, and the write would raise mid-turn -- escape instead.
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    app_logger = logging.getLogger("app")
    app_logger.handlers[:] = [handler]
    app_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    app_logger.propagate = False
