# Diataxis in this repository

Read `SKILL.md` first, then this file. `SKILL.md` and `references/` are the method; this file is how the method
applies here. Where the two disagree about this repository, this file wins.

## Who writes documentation

One documentation maintainer writes `docs/` and the README. Everyone else describes what changed in a "Docs:" section
of their pull request ([AGENTS.md](../../../AGENTS.md) has the rule and its two exceptions). If you are not the
documentation maintainer, stop here and write that section instead.

## Where each kind lives

| Kind | Where | Documents today |
|---|---|---|
| Tutorial | `docs/TUTORIAL.md`, and `README.md` from "Install" to "Name people" | `docs/TUTORIAL.md` (the demo household, safe to repeat), then the README's path for a real install |
| How-to guide | `MIGRATING.md`, `CONTRIBUTING.md`, and task documents in `docs/` | `MIGRATING.md`, `CONTRIBUTING.md`, `docs/RELEASES.md` (install and make a release), `docs/RUNNING.md`, `docs/CAMERAS.md`, `docs/ZONES.md`, `docs/ACTIVITIES.md`, `docs/OBJECTS.md`, `docs/REMOTE.md`, `docs/TAUGHT-STATE.md`, `docs/NAMING.md`, `docs/PLACES.md`, `docs/INGEST.md`, `docs/BLUETOOTH.md`, `docs/NETWORK.md`, `docs/PROFILING.md`, `docs/MEASUREMENT.md` |
| Reference | The declaring file first, then `docs/` | `config/defaults.toml` (settings), `services/services.toml` (services), `mise.toml` (tasks), tool docstrings, `docs/COMPONENTS.md` (every service and state file), `docs/VISION-STREAMS.md`, `docs/STREAMS.md`, `docs/CONTINUUM-GATEWAY.md`, `docs/CONTINUUM-BROKER.md`, `docs/CONTINUUM-STORAGE.md`, `docs/STORAGE.md`, `docs/CONVENTIONS.md` |
| Explanation | `docs/`, one document per area | `MANIFESTO.md`, `docs/ARCHITECTURE.md`, `docs/CONTINUUM.md`, `docs/CONTINUUM-CAPTURE.md`, `docs/CONTINUUM-SDK.md`, `docs/SERVICES.md`, `docs/VISION.md`, `docs/PEOPLE.md`, `docs/DOMAIN-MODEL.md`, `docs/ANSWERS.md`, `docs/DASHBOARD.md`, `docs/SPATIAL.md`, `docs/ROADMAP.md`, `docs/history/` |

[docs/README.md](../../../docs/README.md) is the reader's index. A document that is added, moved, renamed, or
removed changes that index and the table above in the same pull request.

Several documents still mix kinds. Fix one when you are already editing it: move the steps to a how-to guide, the
tables to reference, and the reasons to explanation, and link between them. Don't reorganise for its own sake, and
never add an empty folder or a placeholder page.

## Rules that are specific to this repository

- **Every document describes `main`.** No dated logs, no "previously", and no plans. History that an upgrade needs
  goes in `MIGRATING.md`; planned work is an issue on the board.
- **Reference lives where the thing is declared.** Settings are documented in `config/defaults.toml`, services in
  `services/services.toml`, tasks in `mise.toml`, and a tool in its own docstring. A document links to them and never
  copies their lists.
- **`MIGRATING.md` and the notes in `docs/RELEASES.md` arrive with the code.** Their authors write the steps; the
  documentation maintainer edits them afterwards for form and never changes what a step does without checking it.
- **Module docstrings and code comments belong to the code's author.** Change one only when a document moves and a
  comment points at the old place.
- **Check every fact against the tree.** Every command, path, setting, task, and service name in a document must
  exist: `grep` for it, or run the command's `--help`, before you write it. A link to a heading must match the
  heading. If you cannot confirm something, leave it out.
- **A moved or renamed section takes its links with it.** Search the whole repository (`docs/`, the root documents,
  `src/`, `tools/`, `tests/`, `frontend/`, `continuum/`, `config/`, and `services/`) for the old path and anchor.

## House style

- Concise. Say it once, in the one place it belongs, and link to it from elsewhere. If in doubt, drop it.
- Plain words and the present tense. "The broker drops the frame", not "the frame will be dropped".
- No em dashes or en dashes anywhere. Use a hyphen, a comma, a colon, or parentheses, and "to" for a range.
- The Oxford comma, always.
- Wrap prose at 120 columns. Tables and links may run longer.
- Sentence-case headings. A how-to guide's title names the reader's goal; a reference title names the thing.
- Commands go in fenced `sh` blocks, exactly as they are typed.

## This repository is public

[docs/CONVENTIONS.md](../../../docs/CONVENTIONS.md) is the full list. In documentation that means:

- No names of real people or pets. Examples use Alice, Dana, and Sam, and a pet called Rex.
- No live content: no transcript lines, captions, journal entries, or answers, and no screenshot of a real room,
  person, or data. Screenshots come from `mise run demo`.
- No addresses, floor plans of a real home, or network names.
- No hostnames, network addresses, device IDs, or home-directory paths. Say "the Linux machine" or "an Apple Silicon
  Mac", and use `192.0.2.x`, `example.com`, and `example-tailnet.ts.net`.
- No secrets, real or expired.
- No agent, worker, or session names, and no word about who or what made a change.
- Nothing from the earlier private tracker.
- Never skip the hooks with `--no-verify`.
