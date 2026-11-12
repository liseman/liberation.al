"""Fetch the models declared in defaults.toml (and pack.toml overrides), including Ask and Research.

  pack-models            # fetch anything missing for this platform
  pack-models --list     # show what's declared and what's present
  pack-models --no-ollama

Idempotent: models already present are skipped. In a checkout, `mise run models` runs it. Files go where
config.owner_file() puts a model's path: data/models in a checkout, the user's data directory when installed.

A file download that was cut off carries on from where it stopped on the next run (Hugging Face and Ollama do the
same for theirs), and each shows how far it is.

A gated Hugging Face repository this machine can't read yet doesn't stop the run: everything else is fetched, the
gated ones are listed at the end with how to get access, and the exit status is 1. An `optional` one is different:
it is tried with whatever Hugging Face login this machine already has, and a refusal only prints a line saying it
isn't available here, because the install works without it. An entry marked `only_without = "<name>"` is fetched
only when that other one could not be had, so an install with the better model does not carry its fallback either.
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
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from typing import Any, NoReturn

from liberational.config import CHECKOUT, load, owner_file, platform_key


def wanted(entry: Mapping[str, Any], key: str) -> bool:
    if "only_on" in entry and key not in entry["only_on"]:
        return False
    if entry.get("needs") == "cuda" and not shutil.which("nvidia-smi"):  # GPU-only models on machines without one
        return False
    return key not in entry.get("skip_on", [])


def sha256(path: pathlib.Path) -> str:
    """A file's SHA-256, read in chunks (a checkpoint can be gigabytes)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _progress(name: str, done: int, total: int | None) -> str:
    megabytes = f"{done / 1e6:,.0f}"
    return (
        f"  {name}: {megabytes} of {total / 1e6:,.0f} MB ({done * 100 // total}%)"
        if total
        else f"  {name}: {megabytes} MB"
    )


def download(name: str, url: str, part: pathlib.Path, every_s: float = 5.0) -> None:
    """Fetches url into `part`, carrying on from what a run that was cut off left there (an HTTP Range request),
    and printing how far it is every few seconds. A server that ignores the range sends the whole file, which
    replaces what was there. Raises on a network error and leaves `part` for the next run."""
    have = part.stat().st_size if part.exists() else 0
    # A named client, as Wireshark sends the IEEE: registry servers may refuse Python's default User-Agent.
    headers = {"User-Agent": "liberational"} | ({"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    except urllib.error.HTTPError as error:
        if error.code != 416 or not have:
            raise
        # Nothing past what's here. The whole file, if the server's length agrees; else what's here is stale.
        size = (error.headers.get("Content-Range") or "").rpartition("/")[2]
        if size == str(have):
            return
        part.unlink()
        return download(name, url, part, every_s)
    with response as r:
        resumed = bool(have) and getattr(r, "status", 200) == 206
        done = have if resumed else 0
        length = r.headers.get("Content-Length")
        total = done + int(length) if length and length.isdigit() else None
        if resumed:
            print(f"  {name}: carrying on from {done / 1e6:,.0f} MB")
        said = time.monotonic()
        with open(part, "ab" if resumed else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - said >= every_s:
                    said = time.monotonic()
                    print(_progress(name, done, total), flush=True)
    if total is not None and done != total:
        raise OSError(f"{name}: the download stopped at {done} of {total} bytes; run it again to carry on")


def fetch_file(name: str, entry: Mapping[str, Any]) -> None:
    dest = owner_file(entry["path"])
    if dest.exists():
        print(f"  {name}: present ({entry['path']})")
        return
    url = entry["url"]
    print(f"  {name}: downloading {url}")
    # The download is kept beside its destination until it is whole and checked, so a run that was cut off carries
    # on from it, and an interrupted run never leaves a half model where the code looks for one.
    work = dest.parent / f".{dest.name}.fetch"
    work.mkdir(parents=True, exist_ok=True)
    part = work / url.rsplit("/", 1)[-1]
    download(name, url, part)

    def refuse(why: str) -> NoReturn:
        shutil.rmtree(work)  # a wrong file must not be carried on from
        sys.exit(f"  {name}: {why}")

    if part.name.endswith(".zip"):
        # One file out of a zip (`member`, its path inside), checked against sha256 when given.
        expected = entry.get("sha256")
        if expected and sha256(part) != expected:
            refuse(f"checksum mismatch for {url}; not installed")
        member = entry.get("member")
        if not member:
            refuse("a zip needs `member`, the file to take from it")
        with zipfile.ZipFile(part) as z:
            try:
                data = z.read(member)
            except KeyError:
                refuse(f"archive did not contain {member}")
        unpacked = work / dest.name
        unpacked.write_bytes(data)
        unpacked.replace(dest)
    elif part.name.endswith((".tar.bz2", ".tar.gz", ".tgz", ".tar.xz", ".tar")) and tarfile.is_tarfile(part):
        # By its content, not only its name: a PyTorch checkpoint can be a file named .tar, kept as it is.
        # Unpack beside the download and move into place last.
        with tarfile.open(part) as t:
            t.extractall(work, filter="data")
        unpacked = work / dest.name
        if not unpacked.exists():
            refuse(f"archive did not contain {dest.name}")
        unpacked.replace(dest)
    else:
        expected = entry.get("sha256")
        if expected and sha256(part) != expected:
            refuse(f"checksum mismatch for {url}; not installed")
        part.replace(dest)
    shutil.rmtree(work)
    print(f"  {name}: done")


def ollama_hint() -> str:
    """How to get Ollama on this machine: mise in a checkout, Homebrew on a Mac, and its own installer on Linux
    (it has no apt package)."""
    if CHECKOUT is not None:
        return "install it with `mise install`"
    if sys.platform == "darwin":
        return "install it with `brew install ollama`"
    return "install it from https://ollama.com/download (one command; it has no apt package)"


THREADS_TAG = re.compile(r"^(?P<base>.+)-t(?P<threads>[1-9][0-9]*)$")


def fetch_ollama(models: Mapping[str, str]) -> None:
    """Pulls each model that's missing. A tag ending in -t<N> is made from the model before it with PARAMETER
    num_thread N ([models.ollama], ask.model, research.model, or journal.page_model)."""
    if not shutil.which("ollama"):
        print(f"  ollama not found on PATH; skipping ({ollama_hint()})")
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


NOT_AVAILABLE = "not available here"  # an optional model this machine's Hugging Face login cannot read


def _not_available(name: str, repo: str, why: str) -> bool:
    """An optional model that could not be fetched: a clear line, and the run goes on and exits as it would have.
    False, like any other miss, so a fallback declared with `only_without` is still fetched; unlike a required gated
    repository, the caller doesn't collect it, so the exit status doesn't change. Nothing here asks for, prints, or
    stores a token: the login is whatever huggingface_hub already has."""
    print(f"  {name}: {NOT_AVAILABLE} ({repo} {why}); the install works without it")
    return False


def fetch_hf(name: str, entry: Mapping[str, Any]) -> bool:
    """Fetches one repository. False when it's gated and this machine has no access yet (no login, or a login whose
    account hasn't accepted the license or been approved): the caller goes on and reports it at the end. An
    `optional` repository is never a failure: it is simply not available here."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        print(f"  {name}: huggingface_hub not installed; it will download on first use")
        return True
    patterns = {key: entry[key] for key in ("allow_patterns", "revision") if key in entry}
    if "path" in entry:  # a local copy under data/models, like the files above
        dest = owner_file(entry["path"])
        if (dest / "config.json").exists():
            print(f"  {name}: present ({entry['path']})")
            return True
        print(f"  {name}: {entry['repo']} -> {entry['path']}")
        patterns["local_dir"] = dest
    else:
        print(f"  {name}: {entry['repo']}")
    try:
        snapshot = snapshot_download(entry["repo"], **patterns)
    except GatedRepoError:
        if entry.get("optional"):
            return _not_available(name, entry["repo"], "is gated, and this machine's Hugging Face login has no access")
        print(f"  {name}: gated, and this machine has no access to it yet; skipped")
        return False
    except HfHubHTTPError as error:
        # A gated or private repository is also a 401 or 403 to a login that hasn't been approved, and Hugging Face
        # reports a missing credential as "repository not found".
        if entry.get("optional"):
            return _not_available(
                name, entry["repo"], f"refused this machine's login with HTTP {error.response.status_code}"
            )
        raise
    if entry.get("sha256"):
        verify_hf(name, entry, pathlib.Path(snapshot))
    return True


def verify_hf(name: str, entry: Mapping[str, Any], snapshot: pathlib.Path) -> None:
    """Checks a fetched repository's files against the entry's `sha256` table (file name to digest). A file that
    doesn't match is removed from the snapshot, so nothing loads it, and the run stops: a copy of a model from
    another source than its publisher is only the same model while its weights are the same bytes."""
    for file, expected in (entry.get("sha256") or {}).items():
        path = snapshot / file
        found = sha256(path) if path.is_file() else None
        if found != expected:
            path.unlink(missing_ok=True)
            why = (
                f"{name}: {entry['repo']} {file} has checksum {found or 'none (the file is missing)'}, expected "
                f"{expected}; removed it. The repository's content isn't what this configuration was written for."
            )
            if entry.get("optional"):
                _not_available(name, entry["repo"], "is not the file this configuration pins, so it was removed")
                continue
            sys.exit(why)


def gated_help(gated: Mapping[str, str]) -> str:
    """What to do about the gated repositories that were skipped (name to repo), as the run's last lines."""
    lines = ["not fetched (gated on Hugging Face):"]
    lines += [f"  {name}: https://huggingface.co/{repo}" for name, repo in gated.items()]
    lines += [
        "Sign in to Hugging Face, accept the license on each page above (some are approved by hand, which can take",
        "a while), then set HF_TOKEN to a token of that account (or run `uvx --from huggingface_hub hf auth login`)",
        "and run `pack-models` again. Everything else was fetched.",
    ]
    return "\n".join(lines)


def missing() -> list[str]:
    """What this platform wants and isn't here, one line each for a person (`pack-setup --check`): a file, or a
    Hugging Face copy under data/models. A repository downloaded into Hugging Face's own cache is fetched on first
    use, so whether it's here can't be told."""
    config, key = load(), platform_key()
    models = config.get("models", {})
    gone = [
        f"file {name}: {e['path']}"
        for name, e in models.get("files", {}).items()
        if wanted(e, key) and not owner_file(e["path"]).exists()
    ]
    # An optional model (SAM 3 and its fallback, for instance) is never missing: the install works without it, and
    # naming it here would report a gated download this machine cannot be expected to have as a broken install.
    gone += [
        f"{name}: {e['path']} ({e['repo']})"
        for name, e in models.get("hf", {}).items()
        if wanted(e, key)
        and not e.get("optional")
        and "path" in e
        and not (owner_file(e["path"]) / "config.json").exists()
    ]
    return gone


def fetch(ollama: bool = True, listing: bool = False) -> dict[str, str]:
    """Fetches everything declared that this platform wants and doesn't have (or, with `listing`, only prints what
    is declared and present). Returns the gated Hugging Face repositories it had to skip, name to repository."""
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

    if listing:
        for role, tag in ollama_models.items():
            print(f"  ollama  {role}: {tag}")
        for name, e in models.get("hf", {}).items():
            state = (
                f" -> {e['path']} [{'present' if (owner_file(e['path']) / 'config.json').exists() else 'missing'}]"
                if "path" in e
                else ""
            )
            print(
                f"  hf      {name}: {e['repo']}{state}{'' if wanted(e, key) else '  (not for this platform)'}"
                f"{'  (optional)' if e.get('optional') else ''}"
            )
        for name, e in models.get("files", {}).items():
            state = "present" if owner_file(e["path"]).exists() else "missing"
            print(f"  file    {name}: {e['path']} [{state}]{'' if wanted(e, key) else '  (not for this platform)'}")
        return {}

    print("files:")
    for name, e in models.get("files", {}).items():
        if wanted(e, key):
            fetch_file(name, e)
    print("hugging face:")
    gated: dict[str, str] = {}
    fetched: dict[str, bool] = {}
    for name, e in models.get("hf", {}).items():
        if not wanted(e, key):
            continue
        if (source := e.get("only_without")) and fetched.get(source):
            print(f"  {name}: not fetched ({source} is here, and this one is its fallback)")
            continue
        fetched[name] = fetch_hf(name, e)
        if not fetched[name] and not e.get("optional"):
            gated[name] = e["repo"]
    if ollama:
        print("ollama:")
        fetch_ollama(ollama_models)
    return gated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show declared models and whether they're present")
    ap.add_argument("--no-ollama", action="store_true", help="skip Ollama pulls")
    a = ap.parse_args()
    if gated := fetch(ollama=not a.no_ollama, listing=a.list):
        sys.exit(gated_help(gated))


if __name__ == "__main__":
    main()
