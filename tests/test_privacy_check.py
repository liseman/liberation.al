"""tools/privacy_check.py: what the hooks must refuse, what they must let through, and what a public log may show.
Every value here is made up."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "privacy_check.py"
spec = importlib.util.spec_from_file_location("privacy_check", SCRIPT)
assert spec and spec.loader
pc = importlib.util.module_from_spec(spec)
sys.modules["privacy_check"] = pc
spec.loader.exec_module(pc)


def rules(text: str) -> set[str]:
    return {f.rule for f in pc.scan_text("notes.md", text, [])}


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        pytest.param("mail someone.real@gmail.com", "email", id="email"),
        pytest.param("call 907-234-5678", "phone", id="us-phone"),
        pytest.param("call +61412345678", "phone", id="e164-phone"),
        pytest.param("at 1428 Granite Ridge Road", "street-address", id="street"),
        pytest.param("camera at 10.31.7.42", "lan-ip", id="lan"),
        pytest.param("peer 100.101.102.103", "lan-ip", id="tailscale-range"),
        pytest.param("https://box.tail9f8e7d.ts.net/", "tailscale-name", id="tailnet"),
        pytest.param("sensor 3c:84:6a:1f:9b:d2", "mac-address", id="mac"),
        pytest.param("chip 3C846A1F9BD2", "hardware-id", id="bare-mac"),
        pytest.param("GPU-1a2b3c4d-1111-2222-3333-444455556666", "hardware-id", id="gpu"),
        pytest.param("see /home/zebediah/notes", "home-path", id="home"),
        pytest.param("see /Users/zebediah/notes", "home-path", id="mac-home"),
        pytest.param("Speaker 4: back at nine", "transcript-line", id="transcript"),
        pytest.param("orca_session_id:0f0f-aaaa", "session-id", id="session"),
        pytest.param("github.com/liseman/liberational/issues/9", "private-tracker", id="old-repo"),
        pytest.param("posted in C0ABCDEFGH1", "chat-id", id="channel"),
        pytest.param("copied to .agent-workspaces/someone/notes.md", "agent-workspace", id="workspace"),
    ],
)
def test_a_private_value_is_found(text, rule):
    assert rule in rules(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("mail alice@example.com or noreply@github.com", id="example-emails"),
        pytest.param("git clone git@github.com:liseman/liberation.al.git", id="the-public-repo"),
        pytest.param("call +15555550123 or 415-555-0199", id="fictional-phones"),
        pytest.param("rtsps://192.0.2.20:7441 and 127.0.0.1 and 0.0.0.0", id="documentation-addresses"),
        pytest.param("the 10.0.0.0/8 and 192.168.0.0/16 ranges", id="range-names"),
        pytest.param("version 1.2.3.4 of 2026.11.9.1", id="versions"),
        pytest.param("mac aa:bb:cc:dd:ee:01 or 02:00:00:00:00:0a or f0:9f:c2:00:00:01", id="made-up-macs"),
        pytest.param("from 10:00:00-10:00:30 they talked", id="time-range"),
        pytest.param("my-machine.example-tailnet.ts.net:8443", id="example-tailnet"),
        pytest.param("copy it to /home/user/pack and ~/.config", id="placeholder-home"),
        pytest.param("id 01890A5D-AC96-774B-BCCE-B302099A8057", id="uuid"),
        pytest.param("fixes #12 in liseman/liberation.al", id="new-tracker"),
        pytest.param("Speaker identification runs after diarization", id="prose-about-speakers"),
    ],
)
def test_a_placeholder_is_not_a_finding(text):
    assert rules(text) == set()


def denylist(tmp_path: Path, body: str) -> list:
    file = tmp_path / "deny.txt"
    file.write_text(body)
    return pc.load_denylist(file)


def test_deny_terms_match_whole_words_in_any_case_and_cs_terms_only_exactly(tmp_path):
    deny = denylist(
        tmp_path, "# a comment\n[credential]\nCorrect Horse\n[host]\ncs:Pebble\n[address]\nre:\\b77 Nowhere Rd\\b\n"
    )
    text = "correct horse threw a pebble near 77 Nowhere Rd, said Pebble. Pebbles, Correct Horses."
    found = [(f.rule, f.text) for f in pc.scan_text("a.md", text, deny, rules=False)]
    assert sorted(found) == [
        ("denylist:address", "77 Nowhere Rd"),
        ("denylist:credential", "correct horse"),
        ("denylist:host", "Pebble"),
    ]


def test_a_deny_term_right_after_an_escape_in_source_text_is_found(tmp_path):
    deny = denylist(tmp_path, "[credential]\nhunter2long\n")
    source = 'src = "user: viewer\\nhunter2long"'  # as it sits in a .py file: a backslash and an n, then the term
    assert [f.text for f in pc.scan_text("t.py", source, deny, rules=False)] == ["hunter2long"]


def test_a_missing_deny_list_is_empty_not_an_error(tmp_path):
    assert pc.load_denylist(tmp_path / "absent.txt") == []


def test_a_media_file_and_a_deny_term_in_a_path_are_findings_whatever_the_content(tmp_path):
    deny = denylist(tmp_path, "[credential]\nhunter2long\n")
    found = pc.scan_blob("docs/hunter2long/room.jpg", b"\xff\xd8\xff\x00binary", deny, pc.Allow())
    assert {f.rule for f in found} == {"media-file", "denylist:credential"}


def test_binary_content_gets_the_deny_list_only(tmp_path):
    deny = denylist(tmp_path, "[credential]\nhunter2long\n")
    blob = b"\x00\x01 pw=hunter2long at 10.31.7.42 \x00"
    assert {f.rule for f in pc.scan_blob("fixture.bin", blob, deny, pc.Allow())} == {"denylist:credential"}


def write_allow(root: Path, body: str) -> pc.Allow:
    (root / pc.ALLOW_FILE).write_text(body)
    return pc.Allow.load(root)


def test_an_allow_entry_excuses_only_its_rule_path_and_match(tmp_path):
    allow = write_allow(
        tmp_path, '[[allow]]\nrule = "lan-ip"\npath = "tests/*"\nmatch = \'^192\\.168\\.1\\.\'\nreason = "made up"\n'
    )
    text = b"a 192.168.1.20 b 10.31.7.42 c 3c:84:6a:1f:9b:d2"
    assert sorted(f.text for f in pc.scan_blob("tests/test_net.py", text, [], allow)) == [
        "10.31.7.42",
        "3c:84:6a:1f:9b:d2",
    ]
    assert len(pc.scan_blob("src/net.py", text, [], allow)) == 3


def test_no_allow_entry_excuses_a_term_on_the_deny_list(tmp_path):
    allow = write_allow(tmp_path, '[[allow]]\nrule = "*"\npath = "*"\nreason = "everything"\n')
    deny = denylist(tmp_path, "[credential]\nhunter2long\n")
    found = pc.scan_blob("tests/test_x.py", b"hunter2long at 10.31.7.42", deny, allow)
    assert [(f.rule, f.text) for f in found] == [("denylist:credential", "hunter2long")]


def test_an_allow_entry_without_a_reason_is_refused(tmp_path):
    (tmp_path / pc.ALLOW_FILE).write_text('[[allow]]\nrule = "email"\npath = "*"\n')
    with pytest.raises(SystemExit, match="needs a reason"):
        pc.Allow.load(tmp_path)


def test_a_skipped_path_still_gets_the_deny_list_by_name_but_not_the_patterns(tmp_path):
    allow = write_allow(tmp_path, 'skip = ["uv.lock"]\n')
    assert pc.scan_blob("uv.lock", b"version = 10.4.0.35", [], allow) == []


def test_a_redacted_report_shows_no_matched_text_and_no_location_for_deny_terms(capsys):
    findings = [
        pc.Finding("a.md", 3, "email", "someone.real@gmail.com"),
        pc.Finding("secret/place.md", 7, "denylist:credential", "hunter2long", private=True),
    ]
    pc.report(findings, redact=True, fmt="text")
    err = capsys.readouterr().err
    assert "a.md:3: email: [redacted " in err
    assert "1 match(es) on the private deny list" in err
    assert not {"someone.real", "hunter2long", "secret/place.md"} & {w for w in err.replace(":", " ").split()}


def test_the_message_check_ignores_gits_comment_lines_and_fails_on_a_private_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(pc.DENYLIST_ENV, str(tmp_path / "absent.txt"))
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("Fix the gain step\n# On branch at /home/zebediah/dev\n")
    assert pc.main(["--message", str(message)]) == 0
    message.write_text("Fix the gain step seen at 10.31.7.42\n")
    assert pc.main(["--message", str(message)]) == 1


def commit(repo: Path, message: str) -> str:
    """A commit in `repo` by an author with a personal email whose name is on the deny list; its hash."""
    identity = ["-c", "user.name=Zebediah Quince", "-c", "user.email=zebediah@quince-household.net"]
    quiet = ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(["git", "-C", str(repo), *identity, *quiet, "commit", "-q", "-m", message], check=True)
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_the_range_check_reads_messages_and_content_but_never_the_author_or_committer_line(tmp_path, monkeypatch):
    deny = tmp_path / "deny.txt"
    deny.write_text("[credential]\nZebediah Quince\nquince-household.net\n")
    monkeypatch.setenv(pc.DENYLIST_ENV, str(deny))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.md").write_text("one\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.md"], check=True)
    base = commit(repo, "Add a note")
    (repo / "a.md").write_text("one\ntwo\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.md"], check=True)
    clean = commit(repo, "Add a second line")
    assert pc.main(["--range", base, clean, "--no-secrets"]) == 0  # the author's name and email are on every commit
    (repo / "a.md").write_text("one\ntwo\nthree\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.md"], check=True)
    named = commit(repo, "Add the line Zebediah Quince asked for")
    assert pc.main(["--range", clean, named, "--no-secrets"]) == 1  # the same term in a message is still refused


def test_the_install_terms_are_its_secrets_and_never_its_names_hosts_or_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pc.shutil, "which", lambda _name: None)  # no tailscale on PATH
    (tmp_path / "pack.toml").write_text(
        '[owner]\nname = "Alice"\nphone = "+16175550123"\n[assistant]\nname = "Rex"\n'
        '[phone]\npublic_host = "box.tail9f8e7d.ts.net:8443"\nauth_token = "aaaa-bbbb-cccc"\n'
        '[unifi]\nurl = "https://10.31.7.1"\npassword = "hunter2-long"\n'
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        '[[device]]\nname = "kitchen door"\nkind = "rtsp"\n'
        'rtsp = { url = "rtsp://viewer:s3cretpw@10.31.7.42:554/stream" }\nonvif_hosts = ["192.168.100.243"]\n'
    )
    assert pc.install_terms(tmp_path) == {
        "credential": {"aaaa-bbbb-cccc", "hunter2-long", "s3cretpw"},
        "phone": {"+16175550123", "6175550123"},
        "host": set(),  # no camera, router, or public host, and not this machine's name; an address is no phone
    }


def test_the_install_terms_take_the_tailnets_domain_and_addresses_but_no_machine_name(tmp_path, monkeypatch):
    status = {
        "MagicDNSSuffix": "tail9f8e7d.ts.net",
        "Self": {
            "HostName": "box",
            "DNSName": "box.tail9f8e7d.ts.net.",
            "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1"],
        },
        "Peer": {
            "nodekey:ab12": {
                "HostName": "phone",
                "DNSName": "phone.tail9f8e7d.ts.net.",
                "TailscaleIPs": ["100.101.102.104"],
            }
        },
    }

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd == ["/usr/bin/tailscale", "status", "--json"]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(status), stderr="")

    monkeypatch.setattr(pc.shutil, "which", lambda _name: "/usr/bin/tailscale")
    monkeypatch.setattr(pc.subprocess, "run", run)
    assert pc.install_terms(tmp_path)["host"] == {
        "tail9f8e7d.ts.net",
        "100.101.102.103",
        "fd7a:115c:a1e0::1",
        "100.101.102.104",
    }


def test_the_tailnets_domain_on_the_deny_list_finds_every_machine_under_it(tmp_path):
    deny = denylist(tmp_path, "[host]\ntail9f8e7d.ts.net\n")
    found = pc.scan_text("a.md", "open https://box.tail9f8e7d.ts.net:8443/ first", deny, rules=False)
    assert [(f.rule, f.text) for f in found] == [("denylist:host", "tail9f8e7d.ts.net")]


def test_a_new_deny_list_has_an_empty_address_section_and_a_rebuild_keeps_what_was_added_to_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pc.shutil, "which", lambda _name: None)
    out = tmp_path / "cfg" / "deny.txt"
    assert pc.write_denylist(tmp_path, out) == 0
    assert "[address]" in out.read_text().splitlines()
    assert pc.load_denylist(out) == []  # nothing is refused until the owner fills it in
    out.write_text(out.read_text() + "re:\\b77 Nowhere Rd\\b\n")
    assert pc.write_denylist(tmp_path, out) == 0
    found = pc.scan_text("a.md", "deliver to 77 Nowhere Rd", pc.load_denylist(out), rules=False)
    assert [(f.rule, f.text) for f in found] == [("denylist:address", "77 Nowhere Rd")]
    assert out.stat().st_mode & 0o777 == 0o600
