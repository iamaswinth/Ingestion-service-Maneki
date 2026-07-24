"""Fail-fast configuration checks, run once at startup.

Every secret in config.py defaults to `""` so local dev and tests work with no
`.env` at all. The cost is that a misconfigured deployment boots perfectly
happily and only reveals the problem per-request, later, as a 401 or a 500
from somewhere unrelated to the actual mistake — or, worse, doesn't reveal it
at all: an empty `anthropic_api_key` silently disables question generation
rather than erroring.

So the same defaults that make dev pleasant make production dangerous. These
checks resolve that by being environment-gated: `ENVIRONMENT=production` turns
"unset" from a silent default into a refusal to start.

Deliberately not a pydantic validator — those run at import time, which would
make merely importing `app.config` (as every test does) fail on an unset
value. This runs from the lifespan hook instead.
"""

import logging

from .config import settings

logger = logging.getLogger(__name__)

# Environments where an unset secret is a deployment bug rather than a
# developer running without a .env.
_STRICT_ENVIRONMENTS = {"production", "staging"}


class ConfigurationError(RuntimeError):
    pass


def _is_strict() -> bool:
    return settings.environment.strip().lower() in _STRICT_ENVIRONMENTS


def _looks_local(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


def validate_settings() -> None:
    """Raises ConfigurationError if this process would run misconfigured.

    Outside a strict environment this only warns, so nothing here changes the
    local-dev or test experience.
    """
    problems: list[str] = []

    if not settings.internal_service_token:
        # app/auth.py rejects every request when this is empty, so the service
        # would come up "healthy" and then 401 everything — including the
        # voice runtime mid-call.
        problems.append(
            "INTERNAL_SERVICE_TOKEN is not set — every endpoint except /health "
            "would reject all callers."
        )

    if _looks_local(settings.database_url):
        problems.append(
            "DATABASE_URL still points at localhost — almost certainly the dev "
            "default rather than the real database."
        )

    # Conditional: these features are opt-in, and each already has a defined
    # behavior when the key is missing. The bug being caught is enabling the
    # feature and expecting it to work without a key.
    if settings.question_gen_enabled and not settings.anthropic_api_key:
        problems.append(
            "QUESTION_GEN_ENABLED is true but ANTHROPIC_API_KEY is not set — "
            "question generation would be silently skipped on every ingest."
        )
    if settings.sales_script_enabled and not settings.anthropic_api_key:
        problems.append(
            "SALES_SCRIPT_ENABLED is true but ANTHROPIC_API_KEY is not set — "
            "every sales-script generation would fail."
        )

    if not problems:
        return

    if not _is_strict():
        for problem in problems:
            logger.warning("configuration warning (non-strict environment): %s", problem)
        return

    raise ConfigurationError(
        f"Refusing to start in ENVIRONMENT={settings.environment!r}:\n  - "
        + "\n  - ".join(problems)
    )
