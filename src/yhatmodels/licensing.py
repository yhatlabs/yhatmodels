#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Model licence acceptance.

The `yhatmodels` code is Apache-2.0. Model weights are released under their own licence (named in each model's
`config.json` and model card). Weights under an open licence (Apache-2.0, MIT, BSD) download as they are. YHat Labs'
models under another licence (ChakraTS-OS-10M) are accepted once per person with a yhatlabs.com account before the first
download (yhatmodels.account: the browser opens, or a token in YHAT_MODELS_TOKEN). Other repositories' non-open licences
are accepted once per model on this machine: `yhatmodels.accept_license("<repo>")`, or YHATMODELS_ACCEPT_LICENSE=1 (e.g.
in CI) after reading the licence, stored under ~/.cache/yhatmodels/licenses. Weights loaded from a local folder or
already downloaded are not checked: whoever put them there has already obtained them under their licence.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from yhatmodels.catalog import lookup
from yhatmodels.errors import LicenseNotAcceptedError

ENV = "YHATMODELS_ACCEPT_LICENSE"
OPEN_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}  # SPDX ids that need no acceptance step


def needs_acceptance(license_id: str | None) -> bool:
    """Whether weights under this licence need acceptance before download (a missing or unknown licence does)."""
    return (license_id or "").strip().lower() not in OPEN_LICENSES


def _store() -> Path:
    return Path(os.environ.get("YHATMODELS_CACHE", Path.home() / ".cache" / "yhatmodels")) / "licenses"


def _key(model_id: str) -> str:
    entry = lookup(model_id)  # a released model's short name stands for its repository
    return (entry["repo"] if entry else model_id).replace("/", "--")


def license_url(model_id: str) -> str:
    return f"https://huggingface.co/{model_id}/blob/main/LICENSE.md"


def accept_license(model_id: str) -> Path | None:
    """Accept the licence of `model_id` ahead of its first download. A YHat Labs model ("ChakraTS-OS-10M"): the one-time
    acceptance with a yhatlabs.com account (opens the browser); returns the saved token's file, or None when the token
    comes from the environment. Another repository: records here that you have read and accept its licence; returns
    the file the acceptance is stored in."""
    entry = lookup(model_id)
    if entry and needs_acceptance(entry["license"]):
        from yhatmodels.account import ensure_license, stored_token, token_file

        ensure_license(entry["name"], entry["license"])
        return token_file() if stored_token()[1] == "file" else None
    repo = entry["repo"] if entry else model_id
    path = _store() / f"{_key(repo)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": repo, "license": license_url(repo), "accepted_at": time.time()}) + "\n")
    return path


def is_accepted(model_id: str) -> bool:
    return os.environ.get(ENV, "").lower() in ("1", "true", "yes") or (_store() / f"{_key(model_id)}.json").exists()


def ensure_accepted(model_id: str, license_id: str | None = None) -> None:
    if not needs_acceptance(license_id):
        return
    if entry := lookup(model_id):
        from yhatmodels.account import ensure_license

        ensure_license(entry["name"], entry["license"])
        return
    if not is_accepted(model_id):
        raise LicenseNotAcceptedError(
            f"{model_id} is released under its own licence ({license_url(model_id)}). Read it, then accept it once with "
            f'yhatmodels.accept_license("{model_id}") or set {ENV}=1.'
        )
