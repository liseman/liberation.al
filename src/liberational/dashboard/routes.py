"""The dashboard's routes, registered in one table (#61): a method and a path to the function that answers it.

An area's module registers its own routes when it's imported, with the decorators below, and the server looks a
request up here instead of testing its path against every route in turn. `listing()` is every route there is.

A route is one of three kinds:

- exact (`get("/pets")`, `post("/zones")`): one path, found by a dictionary lookup. The same handler can take
  several paths (`get("/zones", "/zones/history")`).
- a pattern (`get_re(r"/clones/(\\d{1,6})/lines")`): a regular expression the whole path must match; the handler gets
  the match.
- a test (`get_when("/people/<name>", lambda p: p.startswith("/people/"))`): any other rule, with the words the
  listing shows for it.

An exact route is looked up first, then the patterns and tests in the order they were registered, so a catch-all
(`/people/<name>`) is registered after the narrower routes it would otherwise take, and never takes an exact path
(`/people/video-thumb.jpg`). Registering an exact path that an earlier pattern or test already matches is refused
when the modules load: which of the two answers would depend on the kind of route, not on the order written.

What a handler is called with:

- GET (and HEAD, which the server answers as GET): `handler(h)`, or `handler(h, match)` for a pattern.
- POST with a JSON body (`post`, `post_when`): `handler(h, body)`. The server reads and parses the body first and
  answers 400 for one that isn't JSON, for an unknown path too, as it always has.
- POST that reads its own body (`post_raw`, `post_raw_when`): `handler(h)`, before any body is read. A module that
  checks who is asking before it reads anything (Settings, devices, Ask) registers its writes this way.
- DELETE (`delete_when`): `handler(h)`.

`h` is the request's handler (dashboard/server.py's `H`): `h._json`, `h._query()`, `h._can_teach()`, and `h.state`.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

type Handler = Callable[..., Any]
type Test = Callable[[str], object]  # a path to something true (a regular expression's match, or True), or falsy


@dataclass(frozen=True)
class Route:
    method: str  # GET, POST, or DELETE
    shown: str  # the path, the pattern, or the words for a test
    handler: Handler
    kind: str  # "exact", "pattern", or "test"
    raw: bool = False  # a POST that reads its own body
    test: Test | None = None  # for a pattern or a test

    @property
    def area(self) -> str:
        """The module that answers it."""
        return self.handler.__module__.rsplit(".", 1)[-1]


class Table:
    def __init__(self) -> None:
        self.exact: dict[tuple[str, str], Route] = {}
        self.ordered: list[Route] = []  # patterns and tests, in the order registered

    def _exact(self, method: str, paths: tuple[str, ...], raw: bool) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            for path in paths:
                if (method, path) in self.exact:
                    raise ValueError(f"{method} {path} is registered twice")
                for route in self.ordered:
                    if route.method == method and route.test is not None and route.test(path):
                        raise ValueError(f"{method} {path} would be taken by {route.shown}, registered before it")
                self.exact[method, path] = Route(method, path, handler, "exact", raw)
            return handler

        return register

    def _ordered(self, method: str, shown: str, kind: str, test: Test, raw: bool) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            self.ordered.append(Route(method, shown, handler, kind, raw, test))
            return handler

        return register

    def get(self, *paths: str) -> Callable[[Handler], Handler]:
        return self._exact("GET", paths, False)

    def get_re(self, pattern: str) -> Callable[[Handler], Handler]:
        return self._ordered("GET", pattern, "pattern", re.compile(pattern).fullmatch, False)

    def get_when(self, shown: str, test: Test) -> Callable[[Handler], Handler]:
        return self._ordered("GET", shown, "test", test, False)

    def post(self, *paths: str) -> Callable[[Handler], Handler]:
        return self._exact("POST", paths, False)

    def post_when(self, shown: str, test: Test) -> Callable[[Handler], Handler]:
        return self._ordered("POST", shown, "test", test, False)

    def post_raw(self, *paths: str) -> Callable[[Handler], Handler]:
        return self._exact("POST", paths, True)

    def post_raw_when(self, shown: str, test: Test) -> Callable[[Handler], Handler]:
        return self._ordered("POST", shown, "test", test, True)

    def delete_when(self, shown: str, test: Test) -> Callable[[Handler], Handler]:
        return self._ordered("DELETE", shown, "test", test, True)

    def find(self, method: str, path: str) -> tuple[Route, object] | None:
        """The route for a request and what its test gave (a pattern's match), or None for a path nobody registered."""
        route = self.exact.get((method, path))
        if route is not None:
            return route, True
        for route in self.ordered:
            if route.method == method and route.test is not None and (found := route.test(path)):
                return route, found
        return None

    def listing(self) -> list[tuple[str, str, str]]:
        """Every route as (method, path or pattern, area), sorted."""
        routes = [*self.exact.values(), *self.ordered]
        return sorted((route.method, route.shown, route.area) for route in routes)


def load(*areas: str) -> None:
    """Import the dashboard modules that register routes, so the table has them: the server names them once."""
    for area in areas:
        importlib.import_module(f"liberational.dashboard.{area}")


TABLE = Table()
get = TABLE.get
get_re = TABLE.get_re
get_when = TABLE.get_when
post = TABLE.post
post_when = TABLE.post_when
post_raw = TABLE.post_raw
post_raw_when = TABLE.post_raw_when
delete_when = TABLE.delete_when
find = TABLE.find
listing = TABLE.listing
