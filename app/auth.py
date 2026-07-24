"""Service-to-service auth for this app's otherwise-unauthenticated
endpoints. This service has no owner/widget auth of its own — the API
gateway and the voice runtime are the only sanctioned callers, and both
present the same shared internal service token (see their own
app/auth/internal.py / config.py for the matching side of this contract).

An unset `internal_service_token` fails closed: every caller gets 401,
never "auth is skipped."
"""

import hmac

from fastapi import Header, HTTPException

from .config import settings


async def require_internal_token(authorization: str = Header(default="")) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    if not settings.internal_service_token or not hmac.compare_digest(
        token, settings.internal_service_token
    ):
        raise HTTPException(status_code=401, detail="Invalid internal service token")
