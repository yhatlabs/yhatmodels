import http.client
import io
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from test_loading import _copy, fake_hub

import yhatmodels
from yhatmodels import ChakraTS, account
from yhatmodels.errors import LicenseNotAcceptedError

REPO = "yhatlabs/ChakraTS-OS-10M"


@pytest.fixture
def api(monkeypatch, tmp_path):
    """The YHat Labs API's licence check on a local port: token -> accepted (or not); unknown tokens get 401."""
    tokens = {"yhm_good": True, "yhm_fresh": False, "api_key_ok": True}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            seen.append((self.path, token))
            license_id = self.path.rsplit("/", 1)[-1]
            if license_id != "chakra-nc-1.0":
                code, body = 404, {"error": {"type": "not_found"}}
            elif token not in tokens:
                code, body = 401, {"error": {"type": "authentication_error"}}
            else:
                code, body = 200, {"license": license_id, "accepted": tokens[token]}
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("YHAT_API_URL", url)
    monkeypatch.setenv("YHAT_SITE_URL", "https://site.test")
    monkeypatch.setenv("YHATMODELS_CACHE", str(tmp_path / "cache"))
    for env in ("YHAT_MODELS_TOKEN", "YHAT_API_KEY", "YHATMODELS_NO_BROWSER"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(account, "_accepted", set())
    monkeypatch.setattr(account, "_in_notebook", lambda: False)
    monkeypatch.setattr(account.sys, "stdin", io.StringIO(""))  # not a terminal unless a test says so
    yield seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def hub(monkeypatch, tiny_model_dir, tmp_path):
    import huggingface_hub

    repo = _copy(tiny_model_dir, tmp_path / "repo", license="chakra-nc-1.0")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hub({REPO: repo}))
    return repo


def test_weights_already_downloaded_are_never_checked(api, monkeypatch, tiny_model_dir, tmp_path):
    import huggingface_hub

    repo = _copy(tiny_model_dir, tmp_path / "repo", license="chakra-nc-1.0")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hub({REPO: repo}, cached={REPO}))
    monkeypatch.setenv("YHAT_API_URL", "http://127.0.0.1:9")  # nothing listens: offline
    assert ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu").config["license"] == "chakra-nc-1.0"
    assert api == []


def test_a_token_whose_account_accepted_downloads(api, hub, monkeypatch):
    monkeypatch.setenv("YHAT_MODELS_TOKEN", "yhm_good")
    assert ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu")
    assert api == [("/v1/licenses/chakra-nc-1.0", "yhm_good")]
    assert not account.token_file().exists()  # a token from the environment is not copied to disk
    ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu")
    assert len(api) == 1  # once per process


def test_an_api_key_works_in_place_of_a_token(api, hub, monkeypatch):
    monkeypatch.setenv("YHAT_API_KEY", "api_key_ok")
    assert ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu")


def test_with_no_terminal_the_error_says_how_to_accept(api, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hub({}))  # nothing may download
    with pytest.raises(LicenseNotAcceptedError) as e:
        ChakraTS.from_pretrained("ChakraTS-OS-10M")
    msg = str(e.value)
    assert "No token found" in msg and "YHAT_MODELS_TOKEN" in msg
    assert "https://site.test/models/license?license=chakra-nc-1.0&model=ChakraTS-OS-10M" in msg


def test_an_account_that_has_not_accepted_is_sent_to_the_page(api, hub, monkeypatch):
    monkeypatch.setenv("YHAT_MODELS_TOKEN", "yhm_fresh")
    with pytest.raises(LicenseNotAcceptedError, match="has not accepted it yet"):
        ChakraTS.from_pretrained("ChakraTS-OS-10M")


def test_a_replaced_token_is_dropped(api, hub):
    account.save_token("yhm_replaced_on_the_website")
    with pytest.raises(LicenseNotAcceptedError, match="no longer valid"):
        ChakraTS.from_pretrained("ChakraTS-OS-10M")
    assert not account.token_file().exists()


def test_an_unreachable_api_is_reported(api, hub, monkeypatch):
    monkeypatch.setenv("YHAT_MODELS_TOKEN", "yhm_good")
    monkeypatch.setenv("YHAT_API_URL", "http://127.0.0.1:9")
    with pytest.raises(LicenseNotAcceptedError, match="Could not reach"):
        ChakraTS.from_pretrained("ChakraTS-OS-10M")


def _get(url):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
    c.request("GET", f"{u.path}?{u.query}")
    r = c.getresponse()
    return r.status, r.getheader("Location")


def test_the_browser_hands_the_token_back(api, hub, monkeypatch):
    monkeypatch.setattr(account, "_in_notebook", lambda: True)  # browser, no terminal to paste in
    answers = []

    def browser(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert url.startswith("https://site.test/models/license?license=chakra-nc-1.0&model=ChakraTS-OS-10M")
        callback, state = q["callback"][0], q["state"][0]
        assert callback.startswith("http://127.0.0.1:")
        answers.append(_get(f"{callback}?token=yhm_planted&state=not-this-one"))  # another page: refused
        answers.append(_get(f"{callback}?token=yhm_good&state={state}"))
        return True

    monkeypatch.setattr(account.webbrowser, "open", browser)
    assert ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu")
    assert answers[0] == (400, None)
    assert answers[1][0] == 302 and answers[1][1].endswith("model=ChakraTS-OS-10M&done=1")
    assert account.token_file().read_text().strip() == "yhm_good"
    assert account.stored_token() == ("yhm_good", "file")


def test_without_a_display_the_token_is_pasted(api, hub, monkeypatch, capsys):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(account.sys, "stdin", Terminal(""))
    monkeypatch.setattr(account, "_has_display", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yhm_good")
    assert ChakraTS.from_pretrained("ChakraTS-OS-10M", device="cpu")
    assert "models/license?license=chakra-nc-1.0" in capsys.readouterr().out
    assert account.stored_token() == ("yhm_good", "file")


def test_accept_license_ahead_of_the_download(api, monkeypatch):
    monkeypatch.setenv("YHAT_MODELS_TOKEN", "yhm_good")
    assert yhatmodels.accept_license("ChakraTS-OS-10M") is None  # the token lives in the environment
    from yhatmodels.cli import main

    assert main(["accept-license", "ChakraTS-OS-10M"]) == 0
