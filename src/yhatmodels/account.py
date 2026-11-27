#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""The one-time license acceptance for YHat Labs' open models, tied to a yhatlabs.com account.

Before the first download of such a model, yhatmodels checks with the YHat Labs API that the person behind a token has
accepted the model's license. With no token, or one whose account has not accepted it yet, it opens yhatlabs.com in the
browser (sign in or create an account, accept the license) and receives the token back on a one-off local address; with
no browser on this computer, the token is pasted. It is then kept in ~/.cache/yhatmodels/token. Weights already downloaded are never checked, so
they work offline.

The token (`yhm_...`) only proves the acceptance: it cannot call the paid API. An API key works in its place.
Environment: YHAT_MODELS_TOKEN (or YHAT_API_KEY) for CI; YHATMODELS_NO_BROWSER=1 to never open a browser;
YHAT_API_URL and YHAT_SITE_URL to point at a test deployment.
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from yhatmodels.errors import LicenseNotAcceptedError

TOKEN_ENV, KEY_ENV, NO_BROWSER_ENV = "YHAT_MODELS_TOKEN", "YHAT_API_KEY", "YHATMODELS_NO_BROWSER"
WAIT_S = 600  # how long the terminal waits for the browser before giving up
_accepted: set[str] = set()  # licenses confirmed in this process


def api_url() -> str:
    return os.environ.get("YHAT_API_URL", "https://api.yhatlabs.com").rstrip("/")


def site_url() -> str:
    return os.environ.get("YHAT_SITE_URL", "https://yhatlabs.com").rstrip("/")


def token_file() -> Path:
    return Path(os.environ.get("YHATMODELS_CACHE", Path.home() / ".cache" / "yhatmodels")) / "token"


def license_page(license_id: str, model: str, **extra: str) -> str:
    return f"{site_url()}/models/license?" + urllib.parse.urlencode({"license": license_id, "model": model, **extra})


def stored_token() -> tuple[str | None, str]:
    """The token to use and where it came from ("env" or "file"): YHAT_MODELS_TOKEN, YHAT_API_KEY, then the saved one."""
    for env in (TOKEN_ENV, KEY_ENV):
        if v := os.environ.get(env, "").strip():
            return v, "env"
    f = token_file()
    return (f.read_text().strip() or None, "file") if f.is_file() else (None, "file")


def save_token(token: str) -> Path:
    f = token_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(token + "\n")
    f.chmod(0o600)
    return f


def check(token: str, license_id: str) -> str:
    """'accepted', 'not_accepted', 'invalid' (unknown or replaced token) or 'unreachable'."""
    req = urllib.request.Request(f"{api_url()}/v1/licenses/{urllib.parse.quote(license_id)}",
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "yhatmodels"})  # fmt: skip
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return "accepted" if json.load(r).get("accepted") else "not_accepted"
    except urllib.error.HTTPError as e:
        return "invalid" if e.code in (401, 403) else "unreachable"
    except (urllib.error.URLError, OSError, ValueError):
        return "unreachable"


def _has_display() -> bool:
    if sys.platform in ("win32", "darwin"):
        return not (os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"))
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _in_notebook() -> bool:
    shell = sys.modules.get("IPython") and getattr(sys.modules["IPython"], "get_ipython", lambda: None)()
    return type(shell).__name__ == "ZMQInteractiveShell"


def _callback_server(state: str, done_url: str) -> tuple[http.server.ThreadingHTTPServer, threading.Event, list[str]]:
    """A one-off server on 127.0.0.1 that takes the token the license page sends back. `state` ties the answer to this
    request, so another page cannot plant a token."""
    got, event = [], threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = 10

        def do_GET(self) -> None:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ok = q.get("state", [""])[0] == state and bool(q.get("token", [""])[0].strip())
            if ok:
                got.append(q["token"][0].strip())
                event.set()
                self.send_response(302)
                self.send_header("Location", done_url)
                self.end_headers()
                return
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"This link does not match the request from your terminal. Run the download again.")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    return server, event, got


def obtain_token(license_id: str, model: str, intro: str) -> str | None:
    """A token from the browser or the terminal; None where neither is possible (CI, scripts, no browser allowed)."""
    tty, notebook = sys.stdin.isatty(), _in_notebook()
    if not (tty or notebook):
        return None
    browser = os.environ.get(NO_BROWSER_ENV, "").strip().lower() not in ("1", "true", "yes") and (notebook or _has_display())
    if not browser:
        if not tty:
            return None
        print(f"{intro}\nOpen this page in a browser (sign in or create an account, then accept):\n\n  "
              f"{license_page(license_id, model)}\n\nthen paste the token it shows.")  # fmt: skip
        try:
            return input("Token: ").strip() or None
        except (EOFError, KeyboardInterrupt):
            return None
    state = secrets.token_urlsafe(16)
    server, event, got = _callback_server(state, license_page(license_id, model, done="1"))
    url = license_page(license_id, model, callback=f"http://127.0.0.1:{server.server_address[1]}/", state=state)
    # The terminal isn't read while the browser has the page: code pasted with the download (a quick start pasted whole)
    # stays queued and runs after it, instead of being taken for a token.
    try:
        print(f"{intro}\nOpening yhatlabs.com in your browser (sign in or create an account, then accept):\n\n  {url}\n\n"
              f"Waiting for the browser (up to {WAIT_S // 60} minutes). If your browser is on another computer, open this "
              f"link there instead, then set YHAT_MODELS_TOKEN to the token it shows:\n\n  {license_page(license_id, model)}\n",
              flush=True)  # fmt: skip
        webbrowser.open(url)
        return got[0] if event.wait(WAIT_S) else None
    except KeyboardInterrupt:
        return None
    finally:
        server.shutdown()
        server.server_close()


def ensure_license(model: str, license_id: str) -> None:
    """Make sure the person running this has accepted `license_id` (the license of `model`), asking them once if not."""
    if license_id in _accepted:
        return
    page = license_page(license_id, model)
    token, source = stored_token()
    status = check(token, license_id) if token else "invalid"
    if status == "accepted":
        _accepted.add(license_id)
        return
    if status == "unreachable":
        raise LicenseNotAcceptedError(
            f"Could not reach {api_url()} to check the license of {model} before its first download. Check the "
            "connection and try again; once downloaded, the model works offline."
        )
    if status == "invalid" and token and source == "file":
        token_file().unlink(missing_ok=True)  # replaced on the website, or from another deployment
    intro = (f"{model} is free for research, personal use, internal evaluation and competitions under its license. It "
             "needs a one-time acceptance with a yhatlabs.com account before the first download; your data stays on "
             "this machine.")  # fmt: skip
    new = obtain_token(license_id, model, intro)
    if new is None:
        why = {"not_accepted": "The account of this token has not accepted it yet.",
               "invalid": "The token is no longer valid." if token else "No token found."}[status]  # fmt: skip
        raise LicenseNotAcceptedError(
            f"{model} needs a one-time license acceptance with a yhatlabs.com account before its first download. {why}\n"
            f"  1. Open {page} (sign in or create an account) and accept the license.\n"
            f"  2. Copy the token the page shows and set it: export {TOKEN_ENV}=<token>\n"
            f"     (or run `yhatmodels accept-license {model}` in a terminal, which does both)."
        )
    status = check(new, license_id)
    if status != "accepted":
        why = {"invalid": "is not valid", "not_accepted": "belongs to an account that has not accepted the license",
               "unreachable": "could not be checked"}[status]  # fmt: skip
        raise LicenseNotAcceptedError(f"That token {why}. Accept the license at {page}, then copy the token it shows.")
    saved = save_token(new)
    _accepted.add(license_id)
    print(f"License accepted. Token saved to {saved}; {model} will download now.")
