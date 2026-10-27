"""
helpers/auth.py

MSAL token management for message-miner.
Exports get_access_token() -> str.

First run: opens browser for interactive sign-in.
Subsequent runs: silent refresh from .token_cache.json.
Refresh tokens expire ~90 days; silent refresh fails gracefully and re-prompts.
"""

import os
from pathlib import Path

import msal
from dotenv import load_dotenv

load_dotenv()

_TENANT_ID = os.environ["AZURE_TENANT_ID"]
_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
_AUTHORITY = f"https://login.microsoftonline.com/{_TENANT_ID}"
_SCOPES = [
    "https://graph.microsoft.com/Chat.Read",
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/User.Read",
]
_REDIRECT_URI = "http://localhost"
_CACHE_PATH = Path(".token_cache.json")


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if _CACHE_PATH.exists():
        cache.deserialize(_CACHE_PATH.read_text(encoding="utf-8"))
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        _CACHE_PATH.write_text(cache.serialize(), encoding="utf-8")


def get_access_token() -> str:
    """
    Return a valid access token for the Microsoft Graph API.

    Attempts silent token acquisition first (from cache). Falls back to
    interactive browser sign-in if no cached token exists or refresh fails.
    """
    cache = _load_cache()
    app = msal.PublicClientApplication(
        _CLIENT_ID,
        authority=_AUTHORITY,
        token_cache=cache,
    )

    result = None
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(_SCOPES, account=accounts[0])

    if not result:
        print("No cached token. Opening browser for sign-in...")
        result = app.acquire_token_interactive(scopes=_SCOPES)

    _save_cache(cache)

    if "access_token" not in result:
        error = result.get("error", "unknown_error")
        desc = result.get("error_description", "no description")
        raise RuntimeError(f"Authentication failed [{error}]: {desc}")

    return result["access_token"]
