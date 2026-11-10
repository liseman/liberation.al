---
name: python-tests
description: Write, review, or fix Python tests that are idiomatic pytest and actually catch bugs. Use whenever writing new tests, adding a regression test for a bug fix, reviewing test code, adding mocks or fixtures, or judging whether an existing suite protects anything. Covers what deserves a test, deriving expected values, mocking at the right level, pytest idioms (fixtures, parametrize, raises, tmp_path, monkeypatch), and when to reach for Hypothesis.
---

# Python tests

A test exists to catch a specific break. Write the tests the behavior needs and
no more: every extra test is maintenance forever, and a suite full of tests that
cannot fail is worse than a small one that can.

## Before writing anything

- Read the neighboring tests, `conftest.py`, and `[tool.pytest.ini_options]` in
  `pyproject.toml`. The repo's conventions (async mode, snapshot library, fixture
  style, test layout, unittest vs pytest) win over anything in this skill.
- Extend the existing test module for that code when it stays readable. Do not
  create a new file per test.
- For a bug fix, write the regression test first and run it against the unfixed
  code. It must fail for the reported reason. An import error, a broken fixture,
  or a typo is not a reproduced bug.

## Every test names the break it catches

Before writing the body, answer: what realistic change to the production code
would make this test fail? If the answer is "none" or "only a deliberate
redesign," do not write the test.

- **Hand-derive expected values.** Use literals and small hand-checked fixtures.
  An expectation computed by the code under test (or its helpers) passes no
  matter what the code does.
- **No change detectors.** Asserting a constant's value, exact log wording, or
  private structure fails on every intentional change and sleeps through bugs.
  Test the behavior that depends on it: not `assert MAX_RETRIES == 5` but "a
  failing call is attempted 5 times and never a 6th."
- **Test your contract, not the library.** Do not test that pydantic validates a
  type, that SQLAlchemy saves a row, or that a dataclass stores its fields.
  Trivial getters, constructors, and forwarding get no tests unless they
  validate, normalize, default, or cause side effects.
- **Behavior over implementation.** Assert on return values, raised exceptions,
  persisted state, emitted output, and calls that cross a real boundary. Not on
  which private helper got called.
- **Cover the edges the code handles.** Empty, zero, None, boundary values,
  malformed input, and the error branch. Also omitted defaults vs explicit
  overrides when both are supported.

## Mocks and doubles

- **Mock the boundary, not what you own.** Network, clock, randomness,
  filesystem outside `tmp_path`, subprocesses, and third-party services. Use the
  real versions of your own classes. If mock setup is more than half the test,
  switch to real components or an in-memory fake.
- **Prefer fakes to mocks.** A tiny in-memory implementation (a dict-backed repo,
  an `httpx.MockTransport`) usually reads better and checks more than a stack of
  `Mock` return values.
- **Autospec every mock.** Use `create_autospec(Thing)` or `patch(...,
  autospec=True)` so wrong signatures and misspelled methods fail instead of
  silently returning another `Mock`.
- **Patch where the name is looked up**, not where it is defined:
  `patch("myapp.service.requests.get")`, not `patch("requests.get")`.
- **Never assert on the mock itself.** An assertion that only fails when the mock
  is removed tests nothing. Assert calls only when the call is the contract
  (arguments, count, or order sent to an external system), and then be specific:
  `assert_called_once_with(...)`, not `assert mock.called`.
- **Mirror real data.** Fake API responses carry the full real shape, not only
  the fields this test reads.
- Inject the clock or random source when you can. Otherwise use whatever the repo
  already uses (`time-machine`, `freezegun`). Never `time.sleep` in a test: wait
  on an event or poll with a timeout.

## Idiomatic pytest

- Plain functions and plain `assert`. No `unittest.TestCase` in a pytest
  codebase, and classes only to group tests that share fixtures.
- No docstrings that restate the test name. Name tests for the behavior:
  `test_login_rejects_expired_token`, not `test_login_2`.
- No logic in tests: no `if`, no `try/except`, no loops over cases. Loops become
  `@pytest.mark.parametrize`; expected exceptions become `pytest.raises`.
- Parametrize variations of one behavior; write separate tests for different
  behaviors. Give non-obvious cases readable ids with `pytest.param(...,
  id="empty-string")`.
- `pytest.raises(ValueError, match=...)` matches with `re.search`, so use
  `re.escape` for literal text and match the distinguishing part of the message,
  not the whole sentence.
- `pytest.approx` for floats. `pytest.warns` for warnings, `caplog` for logging,
  `capsys` for stdout and stderr.
- `tmp_path` for files, `monkeypatch.setenv` and `monkeypatch.setattr` for
  environment and globals. Never touch the real home directory or environment.
- Fixtures are function-scoped by default. Wider scopes only for expensive,
  immutable resources. Use `yield` for teardown, and a factory fixture when tests
  need several variants of an object. Shared fixtures go in the nearest
  `conftest.py`, single-use setup stays inline in the test.
- Imports at module scope. Use `pytest.importorskip` for optional dependencies.
- `xfail` must be `strict=True` with a reason that points at an issue. Register
  custom markers so `--strict-markers` catches typos.
- Async tests follow the repo's config (`asyncio_mode = "auto"` means no marker).
- Snapshot tests (`inline-snapshot`, `syrupy`) only where the repo already uses
  them, and read every generated snapshot before accepting it.

## Property-based tests

When the code has an algebraic shape (encode/decode roundtrip, idempotent
normalizer, sort order, an invariant, or a reference implementation to compare
against), a Hypothesis `@given` test covers more than a list of examples. If
the `property-based-testing` skill (Trail of Bits) is available, load it for this.
Two traps to avoid without it: a property that recomputes the implementation
(`assert add(a, b) == a + b`), and an `assume()` that filters out nearly every
input. Constrain the strategy instead.

## Before finishing

1. Run the new tests and the module they live in. For a bug fix, confirm the
   test failed before the fix and passes after.
2. Mutation check: mentally break the production code (wrong constant, flipped
   branch, missing side effect, empty return, skipped validation). At least one
   test should fail for each. A mutation nothing catches is either an untested
   behavior or a tautological test.
3. Delete any test that could not name its break.

Adapted from obra/superpowers (`writing-good-tests.md`), the FastMCP testing
skill, Microsoft debugpy's pytest skill, and the Trail of Bits property-based
testing skill.
