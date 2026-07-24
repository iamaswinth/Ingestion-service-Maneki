"""Logging + error-tracking bootstrap. Both functions are called once at
the top of app/main.py, before the FastAPI app is constructed.
"""

import json
import logging
import sys

import sentry_sdk

from .config import settings

_RESERVED_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Fields passed via logger.info(..., extra={...}) land on the
        # LogRecord as plain attributes — surface anything non-standard
        # (job_id, tenant_id, chunk counts, ...) directly in the JSON line.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    formatter: logging.Formatter = (
        _JsonFormatter()
        if settings.log_json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # Route uvicorn's own request/error logs through the same JSON handler
    # instead of its default plain-text formatter — structured request logs
    # without a custom middleware.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.propagate = False


def init_sentry() -> None:
    # Fail-open, matching this repo's existing convention for optional
    # integrations (anthropic_api_key, langchain_api_key): an unset DSN is
    # a silent no-op, not an error.
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )
