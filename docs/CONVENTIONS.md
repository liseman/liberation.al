# Conventions for a public repository

This repository is public, and the system it builds records real homes and real people. Everything written here
(code, comments, tests, docs, issues, pull requests, commit messages, branch names, and the project board) is
readable by anyone, forever. These rules apply to people and agents alike.

## What never appears here

- **People.** No names of household members, guests, callers, or pets, and nothing that describes them (routines,
  relationships, health, or what they said or did). Tests and examples use made-up people: Alice, Dana, and Sam, and
  a pet called Rex.
- **Live content.** No transcript lines, captions, journal entries, answers, call content, recordings, frames, or
  screenshots that show a real room, a real person, or real data. A screenshot comes from made-up data, or it's
  left out.
- **Places.** No street addresses, coordinates, floor plans of a real home, room layouts, or network names.
- **Machines.** No hostnames, Tailscale or LAN names and addresses, MAC addresses, serial numbers, device IDs, or
  paths under someone's home directory. Say "the Linux machine" or "an Apple Silicon Mac", and use the documentation
  ranges (`192.0.2.x`, `example.com`, and `example-tailnet.ts.net`) in examples.
- **Secrets.** No keys, tokens, passwords, or camera URLs with credentials, real or expired.
- **Agents and sessions.** No agent, worker, session, role, or orchestrator names, and no session or task IDs. Work
  is described by what it does, never by who or what did it. An agent doesn't say which machine it ran on. The one
  exception is the sign-off below: the exact model, and the team it ran for.
- **The private tracker.** No links to, quotes from, or screenshots of the project's earlier private repository or
  its board.

If you need a real value to explain a bug, describe its shape instead ("a camera name with a space in it", "a
40-minute recording that crosses midnight").

## Issues

- The title says what's wrong or what's wanted, in neutral words: "Positions read slows down as raw samples grow",
  not who noticed it, where, or when.
- The body has three short parts: what happens now, what should happen, and how to tell it's done. Reproduce with
  made-up data. Logs are trimmed to the lines that matter and checked against the list above before pasting.
- One issue is one piece of work. A large outcome is a `goal` issue with sub-issues under it.
- Never paste from the old tracker. Write the item fresh.

## The board

The project board has five columns.

| Column | Meaning | Who moves cards here |
|---|---|---|
| Ideas | Proposed; nobody has looked at it yet | Anyone |
| Vetting | Being discussed: worth doing, and how | Anyone |
| Ready | Agreed and clear enough to pick up | Maintainers only |
| In progress | Being worked on | Whoever picks it up |
| Done | Merged or closed | Closing the issue does it |

- Picking up an issue means assigning it to the GitHub account doing the work and moving its card to In progress.
  That assignment is the claim. No comment says which agent, session, or machine has it; if parallel work needs
  telling apart, name the branch.
- Agents may open issues in Ideas and move their own work to In progress and Done. Only a maintainer moves a card
  to Ready.
- Nobody changes the board's columns, fields, views, or workflows without a maintainer.

## Branches, commits, and pull requests

- Branches are `<area>/<short-description>`, such as `spatial/positions-read`. No person, agent, model, or machine
  names.
- A commit message says what changed and why. It carries no names beyond the Git author and the sign-off, no machine
  names, and no live content.
- A pull request says `Closes #N` (or `Part of #N`), what changed, and which checks ran with their counts.
  Measurements are given as before and after, or as counts, on "the Linux machine" or "the Mac".
- **Agents sign their work**, whatever the tool, with the exact model and whose team it ran for:
  - A commit's last line is `Co-Authored-By: <exact model> (<name>'s team) <the tool's no-reply address>`, such as
    `Co-Authored-By: Claude Opus 5.5 (Alice's team) <noreply@anthropic.com>`. The model is named with its version,
    never by the tool alone ("Claude", "Cursor Agent"). A tool's own trailer is replaced by this one.
  - A pull request's last line is `Written by <exact model> (<name>'s team).`
  - A draft that one agent wrote and another checked and committed is signed with the model that wrote it.
  - `<name>` is the first name of the person the agent works for. That team label is the only such name allowed:
    worker, session, role, and task names stay out.
- Design decisions are recorded as decisions ("the history window is 30 days"), not as who decided them.

## Guardrails

- `mise run hooks` (part of `mise run setup`) turns on the Git hooks. Before every commit, commit message, and
  push they run `tools/privacy_check.py`: a secret scan (gitleaks), patterns for the categories above, and your
  machine's private deny list. That list holds actual secrets only: the credentials and tokens in your config,
  your phone numbers, your tailnet's addresses and domain, and your street address once you add it.
  `tools/privacy_check.py denylist` builds it from your own install, and it never leaves your machine. Never skip
  the hooks with `--no-verify`.
- The same check runs on every push and pull request on GitHub.
- The checks read what you write: files, commit messages, and a pull request's title and description. They don't
  read a commit's author or committer line; your Git name and email are yours to publish.
- A false alarm is fixed by changing the text when that's easy, and otherwise by an entry in `.privacy-allow.toml`
  with a reason. Don't weaken a pattern to get a commit through.
- The hooks don't know names: a person, a pet, a room, a machine, or an agent is a word like any other to them,
  and the deny list never holds one. They also can't read an issue, a comment, or an image. All of those rely on
  you.

## If something private gets out

Don't push a fix on top and move on: the old content stays reachable. Tell a maintainer at once and privately (not
in an issue). They remove it (deleting the comment's edit history, or rewriting the branch and asking GitHub to
purge cached views), and rotate anything that was a secret.
