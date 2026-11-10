#!/usr/bin/env python3
"""Fetch the models declared in config/defaults.toml (and pack.toml overrides), including Ask and Research.

  uv run tools/fetch_models.py            # fetch anything missing for this platform
  uv run tools/fetch_models.py --list     # show what's declared and what's present
  uv run tools/fetch_models.py --no-ollama

Idempotent: models already present are skipped. Usually run via `mise run models`.

A gated Hugging Face repository this machine can't read yet doesn't stop the run: everything else is fetched, the
gated ones are listed at the end with how to get access, and the exit status is 1.
"""

import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from typing import Any

from liberational.config import ROOT, load, platform_key


def wanted(entry: Mapping[str, Any], key: str) -> bool:
    if "only_on" in entry and key not in entry["only_on"]:
        return False
    if entry.get("needs") == "cuda" and not shutil.which("nvidia-smi"):  # GPU-only models on machines without one
        return False
    return key not in entry.get("skip_on", [])


def sha256(path: pathlib.Path) -> str:
    """A file's SHA-256, read in chunks (CUT3R's checkpoint is 3.2 GB)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch_file(name: str, entry: Mapping[str, Any]) -> None:
    dest = ROOT / entry["path"]
    if dest.exists():
        print(f"  {name}: present ({entry['path']})")
        return
    url = entry["url"]
    print(f"  {name}: downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        part = pathlib.Path(tmp) / url.rsplit("/", 1)[-1]
        # A named client, as Wireshark sends the IEEE: registry servers may refuse Python's default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "liberational"})
        with urllib.request.urlopen(req) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if part.name.endswith(".zip"):
            # One file out of a zip (`member`, its path inside), checked against sha256 when given.
            expected = entry.get("sha256")
            if expected and sha256(part) != expected:
                sys.exit(f"  {name}: checksum mismatch for {url}; not installed")
            member = entry.get("member")
            if not member:
                sys.exit(f"  {name}: a zip needs `member`, the file to take from it")
            with zipfile.ZipFile(part) as z:
                try:
                    data = z.read(member)
                except KeyError:
                    sys.exit(f"  {name}: archive did not contain {member}")
            unpacked = pathlib.Path(tmp) / dest.name
            unpacked.write_bytes(data)
            unpacked.replace(dest)
        elif part.name.endswith((".tar.bz2", ".tar.gz", ".tgz", ".tar.xz", ".tar")) and tarfile.is_tarfile(part):
            # By its content, not only its name: GeoCalib's checkpoint is a PyTorch file named .tar, kept as it is.
            # Unpack in the temp dir and move into place last, so an interrupted run never leaves a half model.
            with tarfile.open(part) as t:
                t.extractall(tmp, filter="data")
            unpacked = pathlib.Path(tmp) / dest.name
            if not unpacked.exists():
                sys.exit(f"  {name}: archive did not contain {dest.name}")
            unpacked.replace(dest)
        else:
            expected = entry.get("sha256")
            if expected and sha256(part) != expected:
                sys.exit(f"  {name}: checksum mismatch for {url}; not installed")
            part.replace(dest)
    print(f"  {name}: done")


THREADS_TAG = re.compile(r"^(?P<base>.+)-t(?P<threads>[1-9][0-9]*)$")


def fetch_ollama(models: Mapping[str, str]) -> None:
    """Pulls each model that's missing. A tag ending in -t<N> is made from the model before it with PARAMETER
    num_thread N ([models.ollama], ask.model, research.model, or journal.page_model)."""
    if not shutil.which("ollama"):
        print("  ollama not found on PATH; skipping (install it with `mise install`)")
        return
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if have.returncode != 0:
        print("  Ollama isn't answering; start it (`ollama serve`) and re-run. Skipping.")
        return
    present = {line.split()[0] for line in have.stdout.splitlines()[1:] if line.strip()}
    for role, tag in models.items():
        if tag in present or f"{tag}:latest" in present:
            print(f"  {role}: present ({tag})")
            continue
        derived = THREADS_TAG.match(tag)
        base = derived["base"] if derived else tag
        if base not in present and f"{base}:latest" not in present:
            print(f"  {role}: pulling {base}")
            subprocess.run(["ollama", "pull", base], check=True)
            present.add(base)
        if derived:
            print(f"  {role}: making {tag} ({base} with {derived['threads']} CPU threads)")
            with tempfile.TemporaryDirectory() as tmp:
                modelfile = pathlib.Path(tmp) / "Modelfile"
                modelfile.write_text(f"FROM {base}\nPARAMETER num_thread {derived['threads']}\n")
                subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True)
            present.add(tag)


def fetch_hf(name: str, entry: Mapping[str, Any]) -> bool:
    """Fetches one repository. False when it's gated and this machine has no access yet (no token, or a token whose
    account hasn't accepted the license or been approved): the caller goes on and reports it at the end."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        print(f"  {name}: huggingface_hub not installed; it will download on first use")
        return True
    patterns = {key: entry[key] for key in ("allow_patterns", "revision") if key in entry}
    if "path" in entry:  # a local copy under data/models, like the files above
        dest = ROOT / entry["path"]
        if (dest / "config.json").exists():
            print(f"  {name}: present ({entry['path']})")
            return True
        print(f"  {name}: {entry['repo']} -> {entry['path']}")
        patterns["local_dir"] = dest
    else:
        print(f"  {name}: {entry['repo']}")
    try:
        snapshot_download(entry["repo"], **patterns)
    except GatedRepoError:
        print(f"  {name}: gated, and this machine has no access to it yet; skipped")
        return False
    return True


def gated_help(gated: Mapping[str, str]) -> str:
    """What to do about the gated repositories that were skipped (name to repo), as the run's last lines."""
    lines = ["not fetched (gated on Hugging Face):"]
    lines += [f"  {name}: https://huggingface.co/{repo}" for name, repo in gated.items()]
    lines += [
        "Sign in to Hugging Face, accept the license on each page above (some are approved by hand, which can take",
        "a while), then set HF_TOKEN to a token of that account (or run `uvx --from huggingface_hub hf auth login`)",
        "and run `mise run models` again. Everything else was fetched.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show declared models and whether they're present")
    ap.add_argument("--no-ollama", action="store_true", help="skip Ollama pulls")
    a = ap.parse_args()
    config, key = load(), platform_key()
    models = config.get("models", {})
    ollama_models: dict[str, str] = dict(models.get("ollama", {}))
    for role in ("ask", "research"):
        tag = config.get(role, {}).get("model", "").strip()
        if tag:
            ollama_models[role] = tag
    if tag := config.get("journal", {}).get("page_model", "").strip():
        ollama_models["journal pages"] = tag
    print(f"platform: {key}")

    if a.list:
        for role, tag in ollama_models.items():
            print(f"  ollama  {role}: {tag}")
        for name, e in models.get("hf", {}).items():
            state = (
                f" -> {e['path']} [{'present' if (ROOT / e['path'] / 'config.json').exists() else 'missing'}]"
                if "path" in e
                else ""
            )
            print(f"  hf      {name}: {e['repo']}{state}{'' if wanted(e, key) else '  (not for this platform)'}")
        for name, e in models.get("files", {}).items():
            state = "present" if (ROOT / e["path"]).exists() else "missing"
            print(f"  file    {name}: {e['path']} [{state}]{'' if wanted(e, key) else '  (not for this platform)'}")
        return

    print("files:")
    for name, e in models.get("files", {}).items():
        if wanted(e, key):
            fetch_file(name, e)
    print("hugging face:")
    gated: dict[str, str] = {}
    for name, e in models.get("hf", {}).items():
        if wanted(e, key) and not fetch_hf(name, e):
            gated[name] = e["repo"]
    if not a.no_ollama:
        print("ollama:")
        fetch_ollama(ollama_models)
    if gated:
        sys.exit(gated_help(gated))


if __name__ == "__main__":
    main()
