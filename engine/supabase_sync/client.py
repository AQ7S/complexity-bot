"""Supabase client wrapper — service-role only (engine never uses anon)."""
from __future__ import annotations

from supabase import Client, create_client

from engine.config import settings

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not settings.have_supabase():
            raise RuntimeError(
                "Supabase not configured: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
            )
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    return _client


def reset_client_for_tests() -> None:
    global _client
    _client = None
