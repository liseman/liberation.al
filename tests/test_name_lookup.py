import json
from unittest.mock import create_autospec

import pytest
import requests

from liberational.people import name_lookup as lookup


@pytest.fixture
def web_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(lookup.cfg, "OWNER_ROOT", tmp_path)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "test-secret")


def test_dedicated_credentials_file_is_allowlisted_and_environment_wins(monkeypatch, tmp_path, web_keys):
    (tmp_path / ".env.search").write_text("BRAVE_API_KEY=file-key\nGOOGLE_CSE_CX=engine\nUNRELATED=ignored\n")
    assert lookup.search_credentials() == {"BRAVE_API_KEY": "test-secret", "GOOGLE_CSE_CX": "engine"}
    monkeypatch.delenv("BRAVE_API_KEY")
    assert lookup.search_credentials()["BRAVE_API_KEY"] == "file-key"


def test_web_preserves_nonwikidata_sources_and_refinement(monkeypatch, web_keys):
    rows = [
        {"title": "Dana Example - musician", "url": "https://example.org/music", "description": "Musician"},
        {"title": "Dana at Acme", "url": "https://example.org/team", "description": "Engineer in Toronto at Acme"},
        {"title": "Duplicate", "url": "https://example.org/team/#bio", "description": "Ignored duplicate"},
        {"title": "Unsafe", "url": "javascript:alert(1)", "description": "Engineer"},
    ]
    get = create_autospec(requests.get, return_value=response({"web": {"results": rows}}))
    monkeypatch.setattr(requests, "get", get)
    result = lookup.lookup("Dana Example", "engineer Toronto Acme")
    assert result["provider"] == "Brave Web Search"
    assert [r["url"] for r in result["candidates"]] == ["https://example.org/team", "https://example.org/music"]
    assert result["candidates"][0]["source_title"] == "Dana at Acme"
    assert result["candidates"][0]["source_snippet"] == "Engineer in Toronto at Acme"
    assert result["candidates"][0]["name"] == "Dana Example"
    assert get.call_args.kwargs["params"]["q"] == "Dana Example engineer Toronto Acme"
    assert get.call_args.kwargs["allow_redirects"] is False
    assert get.call_count == 1


def test_google_uses_original_snippet_not_generated_identity(monkeypatch, web_keys):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "google-secret")
    monkeypatch.setenv("GOOGLE_CSE_CX", "engine-id")
    get = create_autospec(
        requests.get,
        return_value=response(
            {"items": [{"title": "Dana - studio", "link": "https://example.org/studio", "snippet": "Local artist"}]}
        ),
    )
    monkeypatch.setattr(requests, "get", get)
    result = lookup.lookup("Dana", "artist")
    assert result["provider"] == "Google Custom Search"
    assert result["candidates"][0]["source_snippet"] == "Local artist"
    assert get.call_args.args[0] == "https://www.googleapis.com/customsearch/v1"
    assert get.call_args.kwargs["params"]["cx"] == "engine-id"


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_web_http_errors_are_actionable(monkeypatch, web_keys, status):
    reply = response({})
    reply.status_code = status
    monkeypatch.setattr(requests, "get", create_autospec(requests.get, return_value=reply))
    result = lookup.lookup("Dana", "")
    assert str(status) in result["error"]
    assert "candidates" not in result


def test_web_timeout_never_leaks_credential(monkeypatch, web_keys):
    monkeypatch.setattr(requests, "get", create_autospec(requests.get, side_effect=requests.Timeout("test-secret")))
    result = lookup.lookup("Dana", "")
    assert "unavailable" in result["error"]
    assert "test-secret" not in result["error"]


@pytest.mark.parametrize("payload", [{"web": {"results": []}}, {"web": {"results": "bad"}}, []])
def test_web_empty_or_malformed(monkeypatch, web_keys, payload):
    monkeypatch.setattr(requests, "get", create_autospec(requests.get, return_value=response(payload)))
    result = lookup.lookup("Dana", "")
    assert result.get("candidates") == [] or "error" in result


def test_missing_credentials_does_not_silently_restrict_to_wikidata(monkeypatch, web_keys):
    monkeypatch.delenv("BRAVE_API_KEY")
    result = lookup.lookup("Dana", "")
    assert "BRAVE_API_KEY" in result["error"]


def response(data):
    result = create_autospec(requests.Response, instance=True)
    result.status_code = 200
    result.__enter__.return_value = result
    result.iter_content.return_value = [json.dumps(data).encode()]
    return result
