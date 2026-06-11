"""Structured logging helpers."""

from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog when available, falling back to stdlib logging."""

    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")
    try:
        import structlog
    except ModuleNotFoundError:
        return

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Any) -> Any:
    """Return a logger with optional contextual binding."""

    try:
        import structlog
    except ModuleNotFoundError:
        return logging.getLogger(name)
    return structlog.get_logger(name).bind(**context)
