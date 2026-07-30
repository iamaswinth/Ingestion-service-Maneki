"""Turns an exception from a generation run into something a dashboard can
show a non-engineer, plus the domain error the graph raises when fact
extraction degraded past the point where a script is worth writing.

app/salescript/store.py's `error` TEXT column is rendered verbatim by the
owner dashboard, so raw `str(exc)` on an Anthropic failure put
"Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error',
...}}" in front of a user whose only useful action is "click retry".
Classify to a sentence that says that, and keep the raw repr as a bounded
suffix so the row is still diagnosable without digging up logs.
"""

import anthropic

_MAX_DETAIL_CHARS = 240


class FactExtractionFloorError(RuntimeError):
    """Too many pages failed fact extraction for the result to be worth
    writing a script from. Its str() is already user-facing prose."""


def _detail(exc: BaseException) -> str:
    raw = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    request_id = getattr(exc, "request_id", None)
    if request_id:
        raw = f"{raw} (request_id={request_id})"
    if len(raw) > _MAX_DETAIL_CHARS:
        raw = raw[: _MAX_DETAIL_CHARS - 1] + "…"
    return raw


def describe_failure(exc: BaseException) -> str:
    """Human-readable sentence for sales_scripts.error. Order matters: more
    specific anthropic exception types must be checked before the generic
    APIStatusError branches, since most of them subclass it (and
    APITimeoutError subclasses APIConnectionError)."""
    if isinstance(exc, FactExtractionFloorError):
        return str(exc)

    if isinstance(exc, anthropic.OverloadedError):
        sentence = (
            "Anthropic's API was temporarily overloaded (HTTP 529). "
            "This is usually transient — try generating again in a few minutes."
        )
    elif isinstance(exc, anthropic.RateLimitError):
        sentence = (
            "Rate limited by the Anthropic API (HTTP 429). Try again shortly, "
            "or lower SALES_SCRIPT_MAX_CONCURRENT_EXTRACTIONS."
        )
    elif isinstance(exc, anthropic.APITimeoutError):
        sentence = "An Anthropic API call timed out. Try again."
    elif isinstance(exc, anthropic.APIConnectionError):
        sentence = (
            "Could not reach the Anthropic API (network error). "
            "Check connectivity and try again."
        )
    elif isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        sentence = (
            f"The Anthropic API rejected our credentials (HTTP {exc.status_code}). "
            "Retrying will not help — check ANTHROPIC_API_KEY."
        )
    elif isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        sentence = (
            f"The Anthropic API returned a server error (HTTP {exc.status_code}). "
            "Usually transient — try again."
        )
    elif isinstance(exc, anthropic.APIStatusError):
        sentence = (
            f"The Anthropic API rejected the request (HTTP {exc.status_code}). "
            "This is a configuration or code problem, not a transient failure."
        )
    elif isinstance(exc, (ValueError, TypeError)):
        sentence = (
            "The model returned output that could not be parsed. "
            "Try again — this is usually a one-off."
        )
    elif isinstance(exc, TimeoutError):
        sentence = "Generation exceeded its time budget and was stopped. Try again."
    else:
        sentence = "Generation failed unexpectedly."

    return f"{sentence} [{_detail(exc)}]"
