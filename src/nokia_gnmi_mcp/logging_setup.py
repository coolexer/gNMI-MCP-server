"""Logging setup for MCP-safe stdio operation."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> logging.StreamHandler:
    """Route all logging to stderr so stdout remains reserved for MCP JSON."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    for name in ("pygnmi", "pygnmi.client", "grpc"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    return handler
