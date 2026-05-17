"""Structured JSON logging for feedback_seg.

Module-level logger configured once on import. Other modules just
import `log` from here.
"""

import json
import logging
import sys


class _JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":    self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }
        # Promote any `extra={}` fields to top-level keys
        for k, v in record.__dict__.items():
            if k in ("ts", "level", "msg", "args", "name", "msg",
                     "levelname", "levelno", "pathname", "filename",
                     "module", "exc_info", "exc_text", "stack_info",
                     "lineno", "funcName", "created", "msecs",
                     "relativeCreated", "thread", "threadName",
                     "processName", "process", "message", "asctime"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = str(v)
        return json.dumps(payload)


def _configure() -> logging.Logger:
    logger = logging.getLogger("feedback_seg")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


log = _configure()
