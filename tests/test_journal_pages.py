import datetime as dt
import json
import re
import subprocess
import types
import urllib.error
import urllib.request
from unittest.mock import create_autospec

import pytest
import requests

from liberational import config as cfg
from liberational import memory_log
from liberational.dashboard import presence
from liberational.listen import revisions
from liberational.memory import history_qa, journal, journal_pages, summarize
from liberational.memory.places import PlaceRef

MON, TUE, WED = dt.date(2023, 11, 13), dt.date(2023, 11, 14), dt.date(2023, 11, 15)
SUN, NEXT_MON = dt.date(2023, 11, 19), dt.date(2023, 11, 20)
DANA = "01a18be2-c1e9-740f-83c3-c4ebf8ea9000"

PARTS_EMPTY = {"night": "", "morning": "", "afternoon": "", "evening": ""}
DAY_REPLY = {
    "in_short": "A quiet day [h1].",
    **{part: "Alice rested [h1]." for part in PARTS_EMPTY},
    "talked_about": [],
    "ate": [],
}
WEEK_REPLY = {"highlights": ["A quiet week [d1]."], "people": ""}
MONTH_REPLY = {"notable": ["A quiet month [d1]."], "trends": [], "changes": []}


def at(day, hh, mm=0):
    return dt.datetime.combine(day, dt.time(hh, mm), dt.UTC).timestamp()


@pytest.fixture
def env(tmp_path, monkeypatch, memory):
    """The pages over the suite's in-memory memory log, in UTC, with the model stubbed by level."""
    monkeypatch.setattr(history_qa, "TZ", dt.UTC)
    monkeypatch.setattr(journal, "model_version", lambda model: "sha256:abc")
    monkeypatch.setattr(journal, "ollama_up", lambda: True)
    journal_pages._TRIES.clear()
    journal_pages._CLOUD_FELL_BACK.clear()
    journal_pages._SO_FAR.clear()
    journal_pages._SO_FAR_BUILDING.clear()
    e = types.SimpleNamespace(
        state=tmp_path, memory=memory, calls=[], replies=dict(day=DAY_REPLY, week=WEEK_REPLY, month=MONTH_REPLY)
    )

    def llm(system, user, model=None, num_predict=1200, temperature=0.2, timeout=300, **kwargs):
        level = next(lv for lv in journal_pages.LEVELS if f"You write one {lv} of" in system)
        part = re.search(r"Now write the (\w+)", system)
        e.calls.append(dict(level=level, part=part and part.group(1), data=json.loads(user), system=system))
        reply = e.replies[level]
        if isinstance(reply, Exception):
            raise reply
        if part and part.group(1) == "summary":  # a day's calls answer from its whole reply, a part at a time
            reply = {"in_short": reply["in_short"]}
        elif part:
            topics, meals = reply.get("talked_about", []), reply.get("ate", [])  # one list for every part, or by part
            topics, meals = (x.get(part.group(1), []) if isinstance(x, dict) else x for x in (topics, meals))
            told = reply[part.group(1)]  # a string: one sentence, citing in its text, as the 4B does
            told = [{"text": told, "cites": []}] if isinstance(told, str) and told else told or []
            reply = {"what_happened": told, "talked_about": topics, "ate": meals}
        return json.dumps(reply), model

    e.llm = create_autospec(summarize.llm, side_effect=llm)
    monkeypatch.setattr(summarize, "llm", e.llm)
    with cfg.override(
        {
            "owner": {"name": "Alice"},
            "models": {"ollama": {"llm": "test-llm"}},
            "journal": {
                "enabled": True,
                "levels": [30, 180, 600, 3600],
                "reconcile_days": 2,
                "model": "",
                "page_model": "",
                "page_batch_at": "04:00",  # not the default 03:00, so the tests below see it read from config
            },
            "timeline": {"enabled": False},
        }
    ):
        yield e


def hour(env, start, text, subjects=(), supersedes=None):
    record = {"ts_start": start, "ts_end": start + 3600, "level": 3600, "text": text}
    if supersedes:
        record["supersedes"] = supersedes
    return env.memory.append(memory_log.JOURNAL, record, subjects=list(subjects))


def pages(env, level=None):
    """Every page and marker written, oldest first."""
    rows = env.memory.records.get(memory_log.PAGES, [])
    return [r for r in rows if not r.get("_purged") and (level is None or r["level"] == level)]


def run(env, now):
    return journal_pages.pages_pass(env.state, now)


def part_calls(env):
    """The model calls for parts of a day, not their summaries."""
    return [c for c in env.calls if c["level"] == "day" and c["part"] != "summary"]


def days_built(env):
    """How many day pages were written: each ends in one summary call."""
    return sum(c["part"] == "summary" for c in env.calls)


def test_a_day_page_has_fixed_sections_links_its_citations_and_lists_its_evidence(env):
    h1 = hour(env, at(TUE, 9), "I worked at the desk with [S3].", subjects=[DANA])
    h2 = hour(env, at(TUE, 12), "I ate soup.")
    env.replies["day"] = {
        "in_short": "Alice worked from home [h1] and had lunch [h2].",
        **PARTS_EMPTY,
        "morning": "Alice worked with [S3] and [S7] [h1][h9].",
        "talked_about": [],
        "ate": ["Lunch: soup [h2]."],
    }
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    t9, t10, t12, t13 = at(TUE, 9), at(TUE, 10), at(TUE, 12), at(TUE, 13)
    assert page["md"] == (
        "# Tuesday 2023-11-14\n"
        "\n## In short\n"
        f"Alice worked from home [h1](#t={t9}) and had lunch [h2](#t={t12}).\n"
        "\n## What happened\n"
        f"- Morning (06:00 to 12:00): Alice worked with [S3] and someone [h1](#t={t9}).\n"
        "\n## Ate\n"
        f"- Lunch: soup [h2](#t={t12}).\n"
        "\n## Evidence\n"
        f"- h1: [09:00](#t={t9}) to [10:00](#t={t10})\n"
        f"- h2: [12:00](#t={t12}) to [13:00](#t={t13})\n"
    )
    assert (page["period"], page["parents"], page["subjects"]) == ("2023-11-14", [h1, h2], [DANA])
    assert page["sections"]["in_short"] == "Alice worked from home and had lunch."
    [morning, afternoon, _summary] = env.calls  # each part with hours, then In short from what they said
    assert [c["part"] for c in env.calls] == ["morning", "afternoon", "summary"]
    assert [(i["label"], i["text"]) for c in (morning, afternoon) for i in c["data"]["inputs"]] == [
        ("h1", "I worked at the desk with [S3]."),
        ("h2", "I ate soup."),
    ]
    assert "third person" in morning["system"] and '"I" is Alice' in morning["system"]


def test_a_day_page_is_written_once_the_day_has_settled_at_four_in_the_morning(env):
    hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 3, 59))
    assert [p for p in pages(env, "day") if p["period"] == "2023-11-14"] == []
    assert env.calls == []
    run(env, at(WED, 4))
    assert env.memory.page("day", "2023-11-14") is not None


def test_nothing_new_means_no_model_call_and_a_day_with_no_hours_gets_a_quiet_marker(env):
    hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 4))
    run(env, at(WED, 5))
    assert len(env.calls) == 2  # the morning's call and the summary's, both from the first pass
    quiet = [p for p in pages(env, "day") if p["period"] == "2023-11-13"]
    assert [(p["status"], p["parents"]) for p in quiet] == [("quiet", [])]
    assert env.memory.page("day", "2023-11-13") is None


def test_a_rebuilt_hour_rewrites_its_day_page_superseding_the_old(env):
    first = hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 4))
    old = env.memory.page("day", "2023-11-14")
    assert old is not None
    again = hour(env, at(TUE, 9), "I read a novel.", supersedes=first)
    run(env, at(WED, 5))
    new = env.memory.page("day", "2023-11-14")
    assert new is not None and (new["supersedes"], new["parents"]) == (old["id"], [again])
    assert part_calls(env)[-1]["data"]["inputs"][0]["text"] == "I read a novel."


def test_a_week_is_written_early_monday_from_its_day_pages_and_follows_a_rewritten_day(env):
    first = hour(env, at(SUN, 10), "I went for a walk.")
    run(env, at(NEXT_MON, 3, 59))
    assert pages(env, "week") == []
    run(env, at(NEXT_MON, 4))
    sunday = env.memory.page("day", "2023-11-19")
    week = env.memory.page("week", "2023-11-13")
    assert sunday is not None and week is not None
    assert week["parents"] == [sunday["id"]]
    assert week["md"].startswith("# Week of 2023-11-13\n\n## Highlights\n- A quiet week [d1](#t=")
    assert f"- d1: [Sunday 2023-11-19](#t={at(SUN, 0)})" in week["md"]
    hour(env, at(SUN, 10), "I went for a long walk.", supersedes=first)
    run(env, at(NEXT_MON, 5))
    sunday_again = env.memory.page("day", "2023-11-19")
    week_again = env.memory.page("week", "2023-11-13")
    assert sunday_again is not None and week_again is not None
    assert (week_again["supersedes"], week_again["parents"]) == (week["id"], [sunday_again["id"]])


def test_a_month_is_written_on_the_first_from_its_day_pages(env):
    last_day = dt.date(2023, 11, 30)
    hour(env, at(last_day, 20), "I cooked dinner.")
    run(env, at(dt.date(2023, 12, 1), 4))
    day = env.memory.page("day", "2023-11-30")
    month = env.memory.page("month", "2023-11-01")
    assert day is not None and month is not None
    assert month["parents"] == [day["id"]]
    assert month["md"].startswith("# November 2023\n\n## Notable events\n- A quiet month [d1](#t=")
    assert env.memory.page("week", "2023-11-27") is None  # that week runs to Sunday 3 December
    month_call = next(c for c in env.calls if c["level"] == "month")
    assert "never by its label" in month_call["system"]  # the first month page called its days "d1" and "d3"


def test_purging_an_hour_takes_its_pages_and_the_next_pass_writes_them_without_it(env):
    h1 = hour(env, at(TUE, 9), "I talked with [S3].", subjects=[DANA])
    hour(env, at(TUE, 12), "I ate soup.")
    run(env, at(WED, 4))
    env.memory.purge([h1], "PersonForgotten")
    assert env.memory.page("day", "2023-11-14") is None  # the broker's cascade took it with its input
    run(env, at(WED, 5))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None and DANA not in page["subjects"]
    assert [i["text"] for i in part_calls(env)[-1]["data"]["inputs"]] == ["I ate soup."]


def test_a_failing_period_is_tried_three_times_in_one_pass_then_waits_for_its_inputs_to_change(env):
    # Tried again at once, not on later passes: with the page model the next batch is a day away.
    hour(env, at(TUE, 9), "I read.")
    env.replies["day"] = RuntimeError("the model broke")
    run(env, at(WED, 4))
    assert len(env.calls) == 3
    run(env, at(WED, 4, 1))
    assert len(env.calls) == 3
    assert [(p["status"], p["period"]) for p in pages(env, "day") if p["period"] == "2023-11-14"] == [
        ("failed", "2023-11-14")
    ]
    env.replies["day"] = DAY_REPLY
    hour(env, at(TUE, 10), "I wrote.")
    run(env, at(WED, 5))
    assert env.memory.page("day", "2023-11-14") is not None


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param({"in_short": "", **PARTS_EMPTY, "talked_about": [], "ate": []}, id="all-empty"),
        pytest.param(
            {"in_short": "[h7]", **PARTS_EMPTY, "night": "  ", "talked_about": [], "ate": []}, id="nothing-left"
        ),
    ],
)
def test_a_reply_with_nothing_in_it_is_a_failure_not_a_page(env, reply):
    hour(env, at(TUE, 9), "I read.")
    env.replies["day"] = reply
    run(env, at(WED, 4))
    assert env.memory.page("day", "2023-11-14") is None
    assert [c["part"] for c in env.calls] == ["morning"] * 3  # nothing to sum up, so no summary call
    assert [p["status"] for p in pages(env, "day") if p["period"] == "2023-11-14"] == ["failed"]


def test_review_chips_come_from_the_day_pages_cited_bullets(env):
    hour(env, at(TUE, 9), "I read.")
    env.replies["day"] = {"in_short": "A day [h1].", **PARTS_EMPTY, "morning": "Reading [h1]."}
    run(env, at(WED, 4))
    a, b = journal.day_bounds(TUE)
    chips = presence.events(env.state, "2023-11-14", a, b)
    assert [(c["label"], c["spans"][0][0]) for c in chips] == [("What happened", at(TUE, 9))]


def test_long_inputs_are_shortened_until_the_prompt_and_reply_fit():
    payload = {"inputs": [{"label": "h1", "when": "", "text": "word " * 4000}], "counts": ""}
    user = journal_pages.fit("system", payload, 1000, "a-model-never-measured", summarize.NUM_CTX)
    assert summarize.prompt_tokens("system", user, "a-model-never-measured") + 1000 + 128 <= summarize.NUM_CTX
    text = json.loads(user)["inputs"][0]["text"]
    assert text.startswith("word word") and text.endswith(" ...")


@pytest.mark.parametrize(
    ("level", "day", "start", "settles"),
    [
        pytest.param("day", TUE, TUE, at(WED, 4), id="day"),
        pytest.param("week", SUN, MON, at(NEXT_MON, 4), id="week-ends-sunday"),
        pytest.param("month", dt.date(2023, 12, 31), dt.date(2023, 12, 1), at(dt.date(2024, 1, 1), 4), id="december"),
        pytest.param("month", dt.date(2024, 2, 29), dt.date(2024, 2, 1), at(dt.date(2024, 3, 1), 4), id="leap-feb"),
    ],
)
def test_each_period_settles_four_hours_after_it_ends(env, level, day, start, settles):
    assert journal_pages.period_start(level, day) == start
    assert not journal_pages.settled(level, start, settles - 1)
    assert journal_pages.settled(level, start, settles)


def test_owner_corrected_days_are_considered_however_old():
    old = dt.date(2023, 10, 2)
    assert old in journal_pages.candidates("day", WED, 2, corrected=[old])
    assert old not in journal_pages.candidates("day", WED, 2)


# ---- the opt-in cloud model for week and month pages ----------------------------------------------------
GATEWAY = "http://gateway.test/v1/chat/completions"
CLOUD = {
    "journal": {"cloud_pages": True, "cloud_agent": "openclaw/journal"},
    "agent": {"url": GATEWAY, "token": "tok", "openclaw_config": ""},
}


def gateway_reply(content, status=200):
    """A chat completion as an OpenAI-compatible gateway answers one."""
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
    r.url = GATEWAY
    return r


@pytest.fixture
def gateway(env, monkeypatch):
    g = types.SimpleNamespace(sent=[], reply=None)

    def post(url, data=None, json=None, **kwargs):
        g.sent.append(dict(url=url, body=json, headers=kwargs.get("headers")))
        if isinstance(g.reply, Exception):
            raise g.reply
        return g.reply

    monkeypatch.setattr(journal_pages.requests, "post", create_autospec(requests.post, side_effect=post))
    return g


def test_cloud_pages_are_off_by_default_and_every_page_is_local(env, gateway):
    hour(env, at(SUN, 10), "I went for a walk.")
    run(env, at(NEXT_MON, 4))
    week = env.memory.page("week", "2023-11-13")
    assert week is not None and week["producer"]["model"] == "test-llm"
    assert gateway.sent == []


@pytest.mark.parametrize(
    ("day", "now", "level", "period", "reply"),
    [
        pytest.param(SUN, at(NEXT_MON, 4), "week", "2023-11-13", WEEK_REPLY, id="week"),
        pytest.param(
            dt.date(2023, 11, 30), at(dt.date(2023, 12, 1), 4), "month", "2023-11-01", MONTH_REPLY, id="month"
        ),
    ],
)
def test_with_the_switch_on_week_and_month_pages_go_to_the_gateway_and_days_stay_local(
    env, gateway, day, now, level, period, reply
):
    hour(env, at(day, 10), "I went for a walk.")
    gateway.reply = gateway_reply("Here is the page:\n```json\n" + json.dumps(reply) + "\n```")
    with cfg.override(CLOUD):
        run(env, now)
    page = env.memory.page(level, period)
    assert page is not None and page["producer"]["model"] == "openclaw/journal"
    assert "[d1](#t=" in page["md"]
    [sent] = gateway.sent
    assert (sent["url"], sent["body"]["model"], sent["headers"]) == (
        GATEWAY,
        "openclaw/journal",
        {"Authorization": "Bearer tok"},
    )
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
    assert {c["level"] for c in env.calls} == {"day"}  # the day page under it stayed local


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(requests.ConnectionError("offline"), id="offline"),
        pytest.param(gateway_reply("", status=502), id="gateway-error"),
        pytest.param(gateway_reply("I can't write that."), id="not-a-page"),
    ],
)
def test_a_failing_gateway_falls_back_to_local_and_says_so_once(env, gateway, capsys, failure):
    first = hour(env, at(SUN, 10), "I went for a walk.")
    gateway.reply = failure
    with cfg.override(CLOUD):
        run(env, at(NEXT_MON, 4))
        hour(env, at(SUN, 10), "I went for a long walk.", supersedes=first)
        run(env, at(NEXT_MON, 5))  # the week is written again, and the gateway fails again
    week = env.memory.page("week", "2023-11-13")
    assert week is not None and week["producer"]["model"] == "test-llm"
    assert len(gateway.sent) == 2
    assert capsys.readouterr().err.count("cloud pages are on, but the cloud model failed") == 1


def test_the_switch_on_without_a_gateway_writes_locally_and_says_so_once(env, gateway, capsys):
    hour(env, at(SUN, 10), "I went for a walk.")
    with cfg.override({**CLOUD, "agent": {"url": ""}}):
        run(env, at(NEXT_MON, 4))
        run(env, at(NEXT_MON, 5))
    week = env.memory.page("week", "2023-11-13")
    assert week is not None and week["producer"]["model"] == "test-llm"
    assert gateway.sent == []
    assert capsys.readouterr().err.count("cloud pages are on, but agent.url is empty") == 1


# ---- today so far -------------------------------------------------------------------------------------------
class Inline:
    """An executor that runs what it's given at once, so a test sees a background build finish before it goes on."""

    def submit(self, fn, *args):
        fn(*args)


@pytest.fixture
def inline(monkeypatch):
    monkeypatch.setattr(journal_pages, "_SO_FAR_POOL", Inline())


def test_today_so_far_has_the_day_pages_sections_says_which_hours_it_covers_and_is_never_stored(env):
    hour(env, at(TUE, 9), "I read.")
    hour(env, at(TUE, 12), "I ate soup.")
    page = journal_pages.so_far(env.state, TUE, at(TUE, 15, 30))
    assert page["status"] == "ready"
    assert page["coverage"] == {"through": at(TUE, 13), "hours": 2, "elapsed": 15}
    assert page["md"].startswith(
        "# Tuesday 2023-11-14, so far\n"
        "So far, through 13:00: written from the 2 of 15 hours with something in them. The day's page is written after"
        " 04:00 tomorrow.\n\n## In short\n"
    )
    assert pages(env) == []  # the day's journal.page stays the settled one


def test_today_is_built_in_the_background_then_reused_until_an_hour_lands_or_it_is_old(env, inline):
    hour(env, at(TUE, 9), "I read.")
    now = at(TUE, 15)
    assert journal_pages.today_page(env.state, now)[0] == 202  # pending: the request never waits on the model
    code, body = journal_pages.today_page(env.state, now + 60)
    assert (code, body["status"], days_built(env)) == (200, "ready", 1)
    assert journal_pages.today_page(env.state, now + 120)[0] == 200
    assert days_built(env) == 1  # opened again: the same page, no model call
    hour(env, at(TUE, 15), "I wrote.")
    code, body = journal_pages.today_page(env.state, now + 180)
    assert (code, body["status"]) == (202, "pending") and body["md"].startswith("# Tuesday")  # the last one meanwhile
    assert days_built(env) == 2
    later = now + 180 + journal_pages.TODAY_KEEP_S
    assert journal_pages.today_page(env.state, later)[0] == 202
    assert days_built(env) == 3


def test_a_written_day_page_is_what_today_answers_with(env, inline):
    hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 4))
    code, body = journal_pages.today_page(env.state, at(WED, 10), TUE)
    assert (code, body["status"]) == (200, "settled")
    assert body["md"].startswith("# Tuesday 2023-11-14\n")
    assert days_built(env) == 1  # only the pass's


def test_only_today_and_an_unwritten_yesterday_are_built_on_request(env, inline):
    hour(env, at(MON, 9), "I read.")
    assert journal_pages.today_page(env.state, at(WED, 10), MON)[0] == 404
    hour(env, at(TUE, 9), "I read.")
    assert journal_pages.today_page(env.state, at(WED, 1), TUE)[0] == 202  # yesterday, before its page settles


def test_a_failed_build_is_tried_again_only_after_a_while(env, inline):
    hour(env, at(TUE, 9), "I read.")
    env.replies["day"] = RuntimeError("the model broke")
    now = at(TUE, 15)
    journal_pages.today_page(env.state, now)
    code, body = journal_pages.today_page(env.state, now + 60)
    assert (code, body["status"]) == (200, "error")
    assert len(env.calls) == 1
    journal_pages.today_page(env.state, now + journal_pages.TODAY_RETRY_S + 60)
    assert len(env.calls) == 2


def test_the_journal_today_route_answers_json_and_refuses_a_bad_day(env, inline, serve_dashboard, monkeypatch):
    from liberational.dashboard import context

    monkeypatch.setattr(context, "STATE", env.state)
    base = serve_dashboard()
    with urllib.request.urlopen(f"{base}/journal/today") as r:
        assert (r.status, json.loads(r.read())["status"]) == (200, "empty")  # nothing written yet today
    with pytest.raises(urllib.error.HTTPError) as bad:
        urllib.request.urlopen(f"{base}/journal/today?day=2023-13-40")
    assert bad.value.code == 400


def test_the_places_line_leaves_out_places_the_owner_only_observed():
    refs = [
        PlaceRef("p-office", "Office", True, 0.0, 10.0, ("Desk",)),
        PlaceRef("p-home", "Home", True, 2.0, 8.0, ("Kitchen",), visited=False),  # a housemate on a home camera
        PlaceRef(None, "", False, 9.0, 9.5, ("Hall", "Desk")),
    ]
    assert journal_pages.place_names(refs) == ["Office (Desk)", "Hall", "Desk"]


# ---- What happened stopped at noon ---------------------------------------------------------------------------
def test_a_page_written_under_an_older_prompt_is_written_again_but_a_quiet_marker_is_not(env, monkeypatch):
    hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 4))
    old = env.memory.page("day", "2023-11-14")
    assert old is not None and days_built(env) == 1
    monkeypatch.setattr(journal_pages, "PROMPT_VERSION", "v-next")
    run(env, at(WED, 5))
    new = env.memory.page("day", "2023-11-14")
    assert new is not None and new["supersedes"] == old["id"]
    assert new["producer"]["prompt_id"] == "journal.page.day.v-next"
    assert days_built(env) == 2  # Monday's quiet marker has no prompt to be out of date with, so it stays


def test_a_part_back_at_its_item_cap_says_so(env, capsys):
    hour(env, at(TUE, 9), "I read.")
    topics = [f"topic {n} [h1]." for n in range(journal_pages.PART_ITEMS)]
    env.replies["day"] = {"in_short": "A day [h1].", **PARTS_EMPTY, "morning": "Reading [h1].", "talked_about": topics}
    run(env, at(WED, 4))
    assert "the morning's Talked about came back at its 4 item cap" in capsys.readouterr().err


def test_a_full_day_keeps_topics_from_every_part_in_the_days_order(env, capsys):
    # The review: four parts of four topics, cut at 8 in part order, lost the evening's.
    for hh in (3, 9, 14, 20):
        hour(env, at(TUE, hh), "I talked.")
    parts = ("night", "morning", "afternoon", "evening")
    env.replies["day"] = {
        "in_short": "A day [h1].",
        **{p: f"Alice talked [h{n}]." for n, p in enumerate(parts, 1)},
        "talked_about": {p: [f"{p} topic {k} [h{n}]." for k in range(1, 5)] for n, p in enumerate(parts, 1)},
    }
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    assert page["sections"]["talked_about"] == [f"{p} topic {k}." for p in parts for k in (1, 2)]
    assert "the day's Talked about kept 8 items of 16 across its parts" in capsys.readouterr().err


def test_what_happened_is_told_a_part_of_the_day_at_a_time_from_the_parts_own_hours(env):
    hour(env, at(TUE, 3), "I slept.")
    hour(env, at(TUE, 14), "I walked to the shop.")
    hour(env, at(TUE, 20), "I cooked.")
    env.replies["day"] = {
        "in_short": "A day [h1].",
        "night": "Alice slept [h1].",
        "morning": "",
        "afternoon": "Alice walked to the shop [h2].",
        "evening": "Alice cooked [h3].",
        "talked_about": [],
        "ate": [],
    }
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    assert page["sections"]["what_happened"] == [
        "Night (00:00 to 06:00): Alice slept.",
        "Afternoon (12:00 to 18:00): Alice walked to the shop.",
        "Evening (18:00 to 24:00): Alice cooked.",
    ]  # in the day's order, an empty part left out
    # a call per part that has hours, each with only its own, then the summary from what the parts said
    assert [(c["part"], [i["text"] for i in c["data"]["inputs"]]) for c in env.calls] == [
        ("night", ["I slept."]),
        ("afternoon", ["I walked to the shop."]),
        ("evening", ["I cooked."]),
        ("summary", ["Alice slept [h1].", "Alice walked to the shop [h2].", "Alice cooked [h3]."]),
    ]


def test_a_sentence_the_schema_stopped_near_its_cap_ends_at_its_last_full_sentence(env):
    # Live, parts once came back at their cap mid-sentence, one ending "in any room,所有"; qwen3:14b's stopped two
    # characters short of it, "present in".
    hour(env, at(TUE, 9), "I read.")
    first = "Alice read a long book by the window in the study all morning while the rain fell outside."
    stopped = (first + " She then sat on the brown sofa by the window" * 5)[: journal_pages.SENTENCE_CHARS - 2]
    env.replies["day"] = {"in_short": "A day [h1].", **PARTS_EMPTY, "morning": [{"text": stopped, "cites": ["h1"]}]}
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    assert page["sections"]["what_happened"] == [f"Morning (06:00 to 12:00): {first}"]


def test_a_parts_sentences_are_cited_from_their_cites_lists(env):
    # qwen3:14b wrote no citations in a part's text; the schema's cites list carries them instead.
    hour(env, at(TUE, 8), "I read.")
    hour(env, at(TUE, 10), "I cooked.")
    env.replies["day"] = {
        "in_short": "A day [h1][h2].",
        **PARTS_EMPTY,
        "morning": [
            {"text": "Alice read.", "cites": ["h1", "h1", "h9"]},  # a repeat and a label the part doesn't have
            {"text": "Then she cooked [h2].", "cites": ["h2"]},  # cited in its text already
            {"text": "  ", "cites": ["h1"]},
        ],
    }
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    t8, t10 = at(TUE, 8), at(TUE, 10)
    assert f"- Morning (06:00 to 12:00): Alice read. [h1](#t={t8}) Then she cooked [h2](#t={t10}).\n" in page["md"]


def test_a_parts_schema_allows_only_its_own_labels_and_needs_one_per_sentence():
    sentence = journal_pages.schema("day", "morning", ["h1", "h2"])["properties"]["what_happened"]["items"]
    assert sentence["required"] == ["text", "cites"]
    assert sentence["properties"]["cites"]["items"]["enum"] == ["h1", "h2"]
    assert sentence["properties"]["cites"]["minItems"] == 1


def test_a_topic_or_meal_listed_by_more_than_one_part_is_listed_once(env):
    hour(env, at(TUE, 9), "I talked about the trip and ate toast.")
    hour(env, at(TUE, 14), "I talked about the trip again and ate soup.")
    env.replies["day"] = {
        "in_short": "A day [h1].",
        **PARTS_EMPTY,
        "morning": "Alice planned [h1].",
        "afternoon": "Alice planned more [h2].",
        "talked_about": ["The trip [h1]."],  # each part's call answers with these
        "ate": ["Toast [h1].", "toast. [h2]"],
    }
    run(env, at(WED, 4))
    page = env.memory.page("day", "2023-11-14")
    assert page is not None
    assert (page["sections"]["talked_about"], page["sections"]["ate"]) == (["The trip."], ["Toast."])
    summary = env.calls[-1]
    assert (summary["data"]["talked_about"], summary["data"]["ate"]) == (["The trip [h1]."], ["Toast [h1]."])


@pytest.mark.parametrize(
    ("text", "most", "kept"),
    [
        pytest.param("One. Two three four.", 100, "One. Two three four.", id="fits"),
        pytest.param(
            "Alice read for an hour. Then she sat on a brown sofa", 30, "Alice read for an hour.", id="sentence"
        ),
        pytest.param("Alice read a long book by the window", 20, "Alice read a long", id="word"),
        pytest.param(
            "Sam said 'Oh here they are.' [h4] Then he left the room and walked to the",
            50,
            "Sam said 'Oh here they are.' [h4]",
            id="quoted",
        ),
        pytest.param(
            "Alice read in the morning; Bob cooked lunch and then sat down by the",
            50,
            "Alice read in the morning.",
            id="clause",
        ),
        # A cut keeps a sentence's citations, and never loses one the full text made
        pytest.param(
            "Alice read [h3]. Bob came [h4]. Cara left after a while [h5].",
            40,
            "Alice read [h3]. Bob came [h4]. [h5]",
            id="cited-before-the-stop",
        ),
        pytest.param(
            "Alice read. [h3] Bob came. [h4] Cara left after a while. [h5]",
            40,
            "Alice read. [h3] Bob came. [h4] [h5]",
            id="cited-after-the-stop",
        ),
        pytest.param(
            "Alice walked outside the long window in the rain [h3][h4]",
            30,
            "Alice walked outside the long [h3][h4]",
            id="word-cut-keeps-citations",
        ),
        pytest.param(
            "Alice ate dinner [h7]! Mia arrived late with friends [h8].", 35, "Alice ate dinner [h7]! [h8]", id="bang"
        ),
    ],
)
def test_text_past_its_limit_is_cut_at_a_sentence_else_a_word(text, most, kept):
    assert journal_pages._trim(text, most) == kept


@pytest.mark.parametrize("level", journal_pages.LEVELS)
def test_the_reply_budget_holds_the_longest_reply_the_schema_allows(level):
    # A day's fixed 1,000 tokens cut its JSON off mid-string, and a month's 1,200 was short of its own limits.
    props = journal_pages.schema(level, labels=["h1", "h12"])["properties"].values()

    def chars(item):  # a cited sentence holds its text and up to its most labels, each like "h12"
        if item["type"] == "object":
            cites = item["properties"]["cites"]
            return item["properties"]["text"]["maxLength"] + cites["maxItems"] * len('"h12",')
        return item["maxLength"]

    longest = sum(chars(p["items"]) * p["maxItems"] if p["type"] == "array" else p["maxLength"] for p in props)
    assert journal_pages.reply_budget(level) >= longest / journal_pages.CHARS_PER_TOKEN + 100


# ---- fit() always ends, and fits (it once spun forever and stopped the journal) ----------------------------
def _hours(n, chars):
    words = ("alice read by the window while bob cooked dinner and the cat slept " * (chars // 60 + 1))[:chars]
    return [
        {"label": f"h{i}", "when": f"{i:02d}:00 to {i + 1:02d}:00", "part": "morning", "text": words} for i in range(n)
    ]


def test_a_day_too_long_for_the_prompt_is_cut_until_it_fits_and_fit_returns():
    # The live shape that hung: 22 to 23 hours of about 1,600 characters each, with a day's whole reply budget.
    payload = {"inputs": _hours(23, 1600), "counts": ""}
    system = journal_pages.system_prompt("day", "morning")
    reply = journal_pages.reply_budget("day")
    user = journal_pages.fit(system, payload, reply, "a-model-never-measured", summarize.NUM_CTX)
    assert summarize.prompt_tokens(system, user, "a-model-never-measured") + reply + 128 <= summarize.NUM_CTX
    texts = [i["text"] for i in json.loads(user)["inputs"]]
    assert len(set(texts)) == 1 and texts[0].endswith(" ...")  # the long hours are cut evenly


def test_inputs_that_fit_are_left_whole_and_short_ones_stay_whole_beside_cut_ones():
    short = _hours(2, 50)
    payload = {"inputs": short, "counts": ""}
    user = journal_pages.fit("system", payload, 1000, "a-model-never-measured", summarize.NUM_CTX)
    assert [i["text"] for i in json.loads(user)["inputs"]] == [h["text"] for h in _hours(2, 50)]
    mixed = {"inputs": [*_hours(1, 40), *_hours(20, 3000)], "counts": ""}
    user = journal_pages.fit("system", mixed, 1800, "a-model-never-measured", summarize.NUM_CTX)
    got = json.loads(user)["inputs"]
    assert got[0]["text"] == _hours(1, 40)[0]["text"] and got[1]["text"].endswith(" ...")


def test_with_no_room_at_all_fit_refuses_rather_than_send_a_prompt_the_model_would_cut():
    payload = {"inputs": _hours(5, 500), "counts": ""}
    with pytest.raises(journal.EmptyReply, match="doesn't fit"):
        journal_pages.fit("x" * 20000, payload, 1000, "a-model-never-measured", summarize.NUM_CTX)


def test_a_cut_below_the_thin_floor_says_so(capsys):
    payload = {"inputs": _hours(23, 1600), "counts": ""}
    journal_pages.fit(
        journal_pages.system_prompt("day", "morning"), payload, 1865, "a-model-never-measured", summarize.NUM_CTX
    )
    assert "a thin page" in capsys.readouterr().err


# ---- the page model's daily batch ---------------------------------------------------------------------------
GiB = 2**30
QWEN3_14B_INFO = {  # /api/show's model_info for qwen3:14b, the fields the guard reads
    "general.architecture": "qwen3",
    "qwen3.block_count": 40,
    "qwen3.attention.head_count": 40,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
}


@pytest.fixture
def big(env, monkeypatch, model_server):
    """The page model "big" (9 GB of weights) set, with a fake Ollama API, one GPU Ollama runs on, and fake services
    (voice and see-captions installed) whose stops and starts are recorded."""
    env.ollama = []  # (path, body) of every Ollama API call besides chat
    env.loaded = []  # /api/ps rows; an unload takes the row, and its size off the GPU
    env.gpus = [{"index": 0, "total": 24 * GiB, "used": 11 * GiB, "apps": {100: 8 * GiB, 200: 3 * GiB}}]
    env.services = []  # ("stop" | "start", names)

    def ollama(path, body=None, timeout=120):
        env.ollama.append((path, body))
        if path == "/api/generate" and body is not None and body["keep_alive"] == 0:
            gone = [m for m in env.loaded if m["name"] == body["model"]]
            env.loaded = [m for m in env.loaded if m["name"] != body["model"]]
            env.gpus[0]["used"] -= sum(m["size"] for m in gone)
        return {
            "/api/ps": {"models": env.loaded},
            "/api/tags": {"models": [{"name": "big", "size": 9 * 10**9}, {"name": "test-llm", "size": 3 * 10**9}]},
            "/api/show": {"model_info": QWEN3_14B_INFO},
        }.get(path, {})

    monkeypatch.setattr(journal_pages, "_ollama", ollama)
    model_server.reply = lambda sent: ollama(sent.path, sent.payload)  # a load or unload (liberational/ollama.py)
    monkeypatch.setattr(journal_pages.nvml, "gpus", lambda: env.gpus)
    monkeypatch.setattr(journal_pages, "_ollama_pids", lambda: {100})  # 200 is another process on the GPU

    env.refuse = set()  # verbs pack-services refuses

    def pack_services(argv, **kwargs):  # the pack-services CLI, as subprocess.run gives back its run
        verb, names = argv[1], argv[2:]
        if verb == "installed":
            return subprocess.CompletedProcess(argv, 0, "voice\nsee-captions\njournal\ndashboard\n", "")
        if verb in env.refuse:
            return subprocess.CompletedProcess(argv, 1, "", "not installed: voice\n")
        env.services.append((verb, names))
        done = "stopped" if verb == "stop" else "running"
        return subprocess.CompletedProcess(argv, 0, "".join(f"pack-{n}.service: {done}\n" for n in names), "")

    monkeypatch.setattr(journal_pages, "subprocess", types.SimpleNamespace(run=pack_services))
    journal_pages._PAGE_MODEL_FELL_BACK.clear()
    with cfg.override({"journal": {"page_model": "big", "page_num_ctx": 8192}}):
        yield env


def chat_models(env):
    """(model, num_ctx, keep_alive) of every chat call."""
    return [(c.kwargs["model"], c.kwargs["num_ctx"], c.kwargs["keep_alive"]) for c in env.llm.call_args_list]


def unloads(env):
    return [body["model"] for path, body in env.ollama if path == "/api/generate" and body["keep_alive"] == 0]


def reloads(env):
    return [
        (path, body["model"], body["options"]) for path, body in env.ollama if body and body.get("keep_alive") == -1
    ]


FOREVER = "2319-02-18T08:56:06.869987559-08:00"  # keep_alive -1, as Ollama reports it
ALWAYS_LOADED = [
    {"name": "test-llm", "size": 3 * 10**9, "context_length": 4096, "expires_at": FOREVER},
    {"name": "test-embed", "size": 10**9, "context_length": 2048, "expires_at": FOREVER},
]
ASKS = {"name": "ask-model", "size": 10**9, "context_length": 16384, "expires_at": "2023-11-15T04:30:00Z"}


def test_the_batch_writes_with_the_page_model_at_its_context_then_unloads_it(big):
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    page = big.memory.page("day", "2023-11-14")
    assert page is not None and page["producer"]["model"] == "big"
    assert set(chat_models(big)) == {("big", 8192, journal_pages.BATCH_KEEP_ALIVE)}
    assert unloads(big) == ["big"]
    assert big.ollama[-1][0] == "/api/generate"  # unloaded after the last page


def test_the_batch_runs_once_a_local_day_from_its_configured_time(big):
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert days_built(big) == 1
    hour(big, at(TUE, 10), "I wrote.")  # the day page is stale now, but today's batch is done
    assert run(big, at(WED, 12))["skipped"] == "batched today"
    run(big, at(dt.date(2023, 11, 16), 3, 59))  # before page_batch_at: not yet, nor counted as Thursday's batch
    assert days_built(big) == 1
    run(big, at(dt.date(2023, 11, 16), 4))
    assert days_built(big) == 2


def test_a_batch_that_stopped_runs_again_on_the_next_pass(big, monkeypatch):
    hour(big, at(TUE, 9), "I read.")
    big.replies["day"] = journal.ollama_health.Unavailable("test-llm", "down")
    assert "stopped" in run(big, at(WED, 4))
    big.replies["day"] = DAY_REPLY
    run(big, at(WED, 4, 1))
    assert big.memory.page("day", "2023-11-14") is not None


def test_a_batch_with_nothing_to_write_loads_nothing(big):
    run(big, at(WED, 4))
    assert big.ollama == [] and big.llm.call_count == 0


WINDOW = ["voice", "see-captions"]  # journal.page_window_services' defaults that the fake has installed


def test_a_page_model_that_doesnt_fit_beside_the_rest_opens_the_sleep_window_and_closes_it_after(big):
    # 9 GiB free: 10.9 GB of model and room for the embedding model don't fit; with Ollama's models unloaded, 13 do.
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = [*ALWAYS_LOADED, ASKS]
    hour(big, at(TUE, 9), "I read.")
    with cfg.override({"models": {"ollama": {"embedding": "test-embed"}}}):
        run(big, at(WED, 4))
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "big"
    assert {m for m, _, _ in chat_models(big)} == {"big"}
    assert big.services == [("stop", WINDOW), ("start", WINDOW)]  # installed ones only; started again after
    assert unloads(big) == ["test-llm", "test-embed", "ask-model", "big"]
    # only the always-loaded ones load again (Ask's would have expired anyway), before their callers start
    assert reloads(big) == [
        ("/api/generate", "test-llm", {"num_ctx": 4096}),
        ("/api/embed", "test-embed", {"num_ctx": 2048}),
    ]
    assert not (big.state / journal_pages.WINDOW_FILE).exists()


def test_a_page_model_that_wont_fit_even_in_the_window_closes_it_and_falls_back(big, capsys):
    big.gpus[0]["used"] = 20 * GiB  # 4 GiB free, 7.7 with Ollama's models gone: the 10.9 GB won't fit
    big.loaded = list(ALWAYS_LOADED)
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert set(chat_models(big)) == {("test-llm", summarize.NUM_CTX, -1)}
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "test-llm"
    assert big.services == [("stop", WINDOW), ("start", WINDOW)]
    assert "even in the sleep window" in capsys.readouterr().err


def test_a_page_model_that_fits_beside_the_rest_stops_nothing(big):
    big.loaded = list(ALWAYS_LOADED)  # 13 GiB free: room for 10.9 GB and the embedding model
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert {m for m, _, _ in chat_models(big)} == {"big"}
    assert big.services == [] and unloads(big) == ["big"] and reloads(big) == []


def test_a_window_a_crash_left_open_is_closed_by_the_next_pass(big):
    window = {"opened": 0.0, "stopped": WINDOW, "models": [{**ALWAYS_LOADED[0], "always": True}]}
    (big.state / journal_pages.WINDOW_FILE).write_text(json.dumps(window))
    run(big, at(WED, 12))
    assert big.services == [("start", WINDOW)]
    assert reloads(big) == [("/api/generate", "test-llm", {"num_ctx": 4096})]
    assert not (big.state / journal_pages.WINDOW_FILE).exists()


def test_past_the_windows_hour_the_rest_is_written_by_the_journals_model_and_the_window_closes(big, monkeypatch):
    monkeypatch.setattr(journal_pages, "WINDOW_MAX_S", -1)  # over as soon as it opens
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = list(ALWAYS_LOADED)
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "test-llm"
    assert big.services == [("stop", WINDOW), ("start", WINDOW)]


def test_a_batch_that_stops_in_its_window_tries_again_without_one(big):
    # Ollama gone mid-batch: the next passes write with what fits, not stop and start the services every minute.
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = list(ALWAYS_LOADED)
    hour(big, at(TUE, 9), "I read.")
    big.replies["day"] = journal.ollama_health.Unavailable("big", "down")
    assert "stopped" in run(big, at(WED, 4))
    big.replies["day"] = DAY_REPLY
    big.gpus[0]["used"] = 15 * GiB  # its models came back
    run(big, at(WED, 4, 1))
    assert big.services == [("stop", WINDOW), ("start", WINDOW)]
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "test-llm"


def test_a_services_refusal_closes_the_window_and_falls_back(big):
    big.refuse = {"stop"}
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = list(ALWAYS_LOADED)
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "test-llm"
    assert big.services == [("start", WINDOW)]
    assert not (big.state / journal_pages.WINDOW_FILE).exists()


def test_a_day_rebuilt_by_hand_never_opens_the_window(big):
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = list(ALWAYS_LOADED)
    hour(big, at(TUE, 9), "I read.")
    journal_pages.rebuild_day(big.state, TUE)
    assert big.services == []
    assert {m for m, _, _ in chat_models(big)} == {"test-llm"}


def shown(env, level, period):
    """The page the Journal tab shows for a period: current, not a marker, and not hidden by a correction."""
    found = [
        r
        for r in memory_log.reviewed(memory_log.current(pages(env)), env.state)
        if r["level"] == level and r["period"] == period and not r.get("status")
    ]
    return found[-1] if found else None


def mark_tv(env, start):
    """The owner marks one transcript line as a TV's, as Review's "This is TV" does."""
    revisions.mark_media(env.state, [{"id": "u1", "ts_start": start, "ts_end": start + 4, "device": "wave3"}])


def test_a_page_a_correction_hides_is_written_again_on_the_next_pass_without_a_window(big):
    # The 2026-11-09 week page: written in Monday's batch, hidden that morning when the owner marked Sunday lines as
    # a TV's, and not written again until Tuesday's batch, so the Journal tab showed the week with no page all day.
    hour(big, at(SUN, 10), "I went for a walk.")
    run(big, at(NEXT_MON, 4))
    assert shown(big, "day", "2023-11-19") is not None and shown(big, "week", "2023-11-13") is not None
    mark_tv(big, at(SUN, 10, 5))
    assert shown(big, "day", "2023-11-19") is None and shown(big, "week", "2023-11-13") is None
    big.gpus[0]["used"] = 15 * GiB  # the page model doesn't fit beside the rest now, and it's day: no window
    big.loaded = list(ALWAYS_LOADED)
    run(big, at(NEXT_MON, 7))
    day, week = shown(big, "day", "2023-11-19"), shown(big, "week", "2023-11-13")
    assert day is not None and week is not None
    assert week["parents"] == [day["id"]]
    assert (day["producer"]["model"], week["producer"]["model"]) == ("test-llm", "test-llm")
    assert big.services == []
    calls = big.llm.call_count
    run(big, at(NEXT_MON, 8))  # nothing corrected since: nothing written again
    assert big.llm.call_count == calls


def test_the_next_batch_writes_a_page_the_journals_model_wrote_in_between_again_with_the_page_model(big):
    hour(big, at(SUN, 10), "I went for a walk.")
    run(big, at(NEXT_MON, 4))
    mark_tv(big, at(SUN, 10, 5))
    big.gpus[0]["used"] = 15 * GiB
    big.loaded = list(ALWAYS_LOADED)
    run(big, at(NEXT_MON, 7))
    run(big, at(NEXT_MON + dt.timedelta(days=1), 4))  # the page model fits in Tuesday's window
    day, week = shown(big, "day", "2023-11-19"), shown(big, "week", "2023-11-13")
    assert day is not None and week is not None
    assert (day["producer"]["model"], week["producer"]["model"]) == ("big", "big")
    assert week["parents"] == [day["id"]]


def test_the_guard_uses_the_size_ollama_reported_in_the_last_batch(big):
    # Measured at 12 GiB loaded (more than the estimate's 10.9 GB), the next batch no longer fits beside the rest.
    hour(big, at(TUE, 9), "I read.")
    big.gpus[0]["used"] = 11 * GiB  # 13 GiB free: room for the estimate and the embedding model, not the measure
    big.loaded = [*ALWAYS_LOADED, {"name": "big", "size": 12 * GiB, "context_length": 8192}]
    run(big, at(WED, 4))  # loaded already: no sizing, but its size is kept at the end
    assert big.services == []
    big.loaded = list(ALWAYS_LOADED)
    big.gpus[0]["used"] = 11 * GiB
    hour(big, at(WED, 9), "I wrote.")
    run(big, at(dt.date(2023, 11, 16), 4))
    assert big.services == [("stop", WINDOW), ("start", WINDOW)]


def test_a_page_model_already_loaded_is_used_without_measuring(big):
    big.loaded = [{"name": "big", "size": 10 * GiB, "context_length": 8192}]
    big.gpus[0]["used"] = 23 * GiB
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert {m for m, _, _ in chat_models(big)} == {"big"}


def test_a_page_model_that_isnt_pulled_falls_back(big):
    with cfg.override({"journal": {"page_model": "missing"}}):
        hour(big, at(TUE, 9), "I read.")
        run(big, at(WED, 4))
    assert {m for m, _, _ in chat_models(big)} == {"test-llm"}


def test_a_page_model_call_that_fails_writes_the_rest_of_the_batch_with_the_journals_model(big):
    fake = big.llm.side_effect

    def llm(system, user, model=None, **kwargs):
        if model == "big":
            raise RuntimeError("all models failed: out of memory")
        return fake(system, user, model=model, **kwargs)

    big.llm.side_effect = llm
    hour(big, at(TUE, 9), "I read.")
    run(big, at(WED, 4))
    assert big.memory.page("day", "2023-11-14")["producer"]["model"] == "test-llm"
    assert [m for m, _, _ in chat_models(big)] == ["big", "test-llm", "test-llm"]  # the morning, again, then In short
    assert unloads(big) == ["big"]


def test_with_no_page_model_pages_are_written_on_every_pass(env):
    hour(env, at(TUE, 9), "I read.")
    run(env, at(WED, 4))
    hour(env, at(TUE, 10), "I wrote.")
    run(env, at(WED, 12))
    assert days_built(env) == 2
    assert {m for m, _, _ in chat_models(env)} == {"test-llm"}


@pytest.mark.parametrize(
    ("ollama", "free"),
    [
        pytest.param({7}, 13, id="ollama-pinned-to-the-busier-gpu"),
        pytest.param(set(), 23, id="ollama-on-none-yet-any-gpu"),
        pytest.param({7, 8}, 23, id="ollama-on-both"),
    ],
)
def test_the_guard_measures_the_gpus_ollama_runs_on(ollama, free):
    gpus = [
        {"total": 24 * GiB, "used": 1 * GiB, "apps": {8: GiB}},
        {"total": 24 * GiB, "used": 11 * GiB, "apps": {7: GiB, 9: GiB}},
    ]
    assert journal_pages.most_free(gpus, ollama) == free * GiB


def test_the_kv_cache_estimate_matches_qwen3_14b_at_8192():
    # 40 layers x 8 KV heads x (128 + 128) x 2 bytes x 8192 tokens: 1.34 GB, about what Ollama added over the file.
    assert journal_pages.kv_bytes(QWEN3_14B_INFO, 8192) == 1_342_177_280
    assert journal_pages.kv_bytes({}, 8192) == 0


def test_a_hybrid_models_kv_cache_counts_only_its_attention_layers():
    # qwen3.8:27b: 65 blocks, full attention every 4th (17 layers), 4 KV heads of 256 + 256.
    info = {
        "qwen35.block_count": 65,
        "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
    }
    assert journal_pages.kv_bytes(info, 8192) == 17 * 4 * 512 * 2 * 8192
