import pytest
from unittest.mock import MagicMock, patch
from requests.auth import HTTPDigestAuth


class TestHandleAuthDigestBranch:
    """Tests for handle_auth covering the digest auth branch at line 47->52."""

    def _make_response(self, auth_header="Digest realm=\"test\"", status_code=401):
        response = MagicMock()
        response.headers = {"www-authenticate": auth_header}
        response.status_code = status_code
        response.request = MagicMock()
        response.request.copy.return_value = MagicMock()
        response.connection = MagicMock()
        inner_response = MagicMock()
        inner_response.history = []
        response.connection.send.return_value = inner_response
        return response

    def test_handle_auth_digest_triggers_on_401(self):
        auth = HTTPDigestAuth("user", "pass")
        auth.num_401_calls = 1
        r = self._make_response(auth_header="Digest realm=\"example.com\"")

        with patch.object(auth, "build_digest_header", return_value="Digest token=abc"):
            with patch("requests.auth.extract_cookies_to_jar"):
                result = auth.handle_auth(r)

        assert result is r.connection.send.return_value
        assert auth.num_401_calls == 2

    def test_handle_auth_skips_digest_on_second_call(self):
        auth = HTTPDigestAuth("user", "pass")
        auth.num_401_calls = 2
        r = self._make_response(auth_header="Digest realm=\"example.com\"")

        result = auth.handle_auth(r)

        assert result is r
        assert auth.num_401_calls == 1

    def test_handle_auth_non_digest_returns_response(self):
        auth = HTTPDigestAuth("user", "pass")
        r = self._make_response(auth_header="Basic realm=\"example.com\"")

        result = auth.handle_auth(r)

        assert result is r
