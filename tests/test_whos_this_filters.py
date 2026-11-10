"""Who's this?: rows already at 100% aren't listed, and query values are URL-decoded."""

from liberational.dashboard import unnamed_groups


def test_a_certain_guess_is_not_asked_about():
    assert unnamed_groups._certain({"band": "probable", "score": 100})
    assert not unnamed_groups._certain({"band": "probable", "score": 99})
    assert not unnamed_groups._certain({"band": "none", "score": 100})


def test_query_values_are_decoded_so_line_keys_with_at_signs_match(dashboard_server):
    handler = object.__new__(dashboard_server.H)  # no socket: _query only reads .path
    handler.path = "/clip?line=respeaker-xvf3800-4-mic-array%401793248055.102&info=1"
    q = dashboard_server.H._query(handler)
    assert q == {"line": "respeaker-xvf3800-4-mic-array@1793248055.102", "info": "1"}
