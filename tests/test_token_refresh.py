"""Regression tests for Homebox token refresh response parsing.

Homebox v0.26 serializes the refresh response as UserAuthTokenDetail with
the token under "raw" (login uses TokenResponse with "token"). The client
must accept both shapes — a regression here logs the user out on every
proactive token refresh.
"""

from __future__ import annotations

import pytest

from homebox_companion.homebox.client import _normalize_token

pytestmark = pytest.mark.unit


class TestNormalizeToken:
    """Test the Bearer prefix normalization helper."""

    def test_strips_bearer_prefix(self) -> None:
        assert _normalize_token("Bearer abc123") == "abc123"

    def test_keeps_bare_token(self) -> None:
        assert _normalize_token("abc123") == "abc123"

    def test_empty_string(self) -> None:
        assert _normalize_token("") == ""


class TestRefreshTokenParsing:
    """Test refresh_token response shape handling (unit-level, mocked HTTP)."""

    @pytest.fixture
    def client(self) -> HomeboxClient:  # type: ignore[name-defined] # noqa: F821
        from homebox_companion.homebox.client import HomeboxClient

        return HomeboxClient(base_url="http://homebox.test/api/v1")

    @pytest.mark.asyncio
    async def test_refresh_raw_shape_v026(self, client, monkeypatch) -> None:
        """Homebox v0.26 refresh returns the token under 'raw' — must parse."""
        import httpx

        raw_token = "NEWTOKEN123"
        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://homebox.test/api/v1/users/refresh"),
            json={"raw": raw_token, "attachmentToken": "att", "expiresAt": "2026-09-16T00:00:00Z"},
        )

        async def fake_get(self, url, headers=None):  # noqa: ARG001
            return response

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        data = await client.refresh_token("OLDTOKEN")
        assert data["token"] == raw_token

    @pytest.mark.asyncio
    async def test_refresh_token_shape_legacy(self, client, monkeypatch) -> None:
        """Older Homebox refresh returns the token under 'token' — still works."""
        import httpx

        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://homebox.test/api/v1/users/refresh"),
            json={"token": "NEWTOKEN456", "expiresAt": "2026-09-16T00:00:00Z"},
        )

        async def fake_get(self, url, headers=None):  # noqa: ARG001
            return response

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        data = await client.refresh_token("OLDTOKEN")
        assert data["token"] == "NEWTOKEN456"

    @pytest.mark.asyncio
    async def test_refresh_bearer_prefix_normalized(self, client, monkeypatch) -> None:
        """'Bearer '-prefixed refresh tokens must be stripped."""
        import httpx

        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://homebox.test/api/v1/users/refresh"),
            json={"raw": "Bearer NEWTOKEN789", "expiresAt": "2026-09-16T00:00:00Z"},
        )

        async def fake_get(self, url, headers=None):  # noqa: ARG001
            return response

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        data = await client.refresh_token("OLDTOKEN")
        assert data["token"] == "NEWTOKEN789"

    @pytest.mark.asyncio
    async def test_refresh_missing_token_raises(self, client, monkeypatch) -> None:
        """A refresh response with no token field must raise, not return empty."""
        import httpx

        from homebox_companion.core.exceptions import HomeboxAuthError

        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://homebox.test/api/v1/users/refresh"),
            json={"expiresAt": "2026-09-16T00:00:00Z"},
        )

        async def fake_get(self, url, headers=None):  # noqa: ARG001
            return response

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        with pytest.raises(HomeboxAuthError):
            await client.refresh_token("OLDTOKEN")
