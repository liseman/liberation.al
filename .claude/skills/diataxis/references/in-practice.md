# Diátaxis in a real repository

> This file is an **extension**, not part of Diátaxis itself. It applies the framework to
> artefacts that software projects actually have. Where it goes beyond diataxis.fr, that is
> an interpretation, offered because "where does the README go?" is the first question
> anyone asks.

## Placing common artefacts

| Artefact | Type | Notes |
|---|---|---|
| **README** | signpost, not a quadrant | See below. Its job is orientation and routing. |
| Installation instructions | how-to | A goal ("get it running"), for someone competent. |
| Quickstart / "first 5 minutes" | tutorial | If it teaches. If it assumes competence, it is a how-to guide. |
| API docs from docstrings | reference | Generated is fine. It is one quadrant, not the docs. |
| CLI `--help` output | reference | Terse reference under a different delivery mechanism. |
| Configuration options | reference | Table. Never prose. |
| CONTRIBUTING.md | mostly how-to | "How to set up a dev environment", "How to submit a PR". Split the culture and rationale into an explanation. |
| ADR / design doc | explanation | Context, decision, alternatives, consequences: explanation by construction. |
| CHANGELOG | reference | A factual record. Never editorialise in it. |
| Release notes | reference + explanation | The list is reference; "why this release matters" is explanation. Keep them visibly separate. |
| Migration guide | how-to | *How to upgrade from v2 to v3.* Link to the explanation of why it changed. |
| Runbook / incident playbook | how-to | Branching, real-world, assumes competence, cannot promise safety. Textbook how-to. |
| Troubleshooting page | how-to | Structure as *symptom → how to fix*, not as a list of error codes. Error codes themselves are reference. |
| FAQ | mixed by nature | An FAQ is a symptom. Each entry belongs somewhere; the FAQ is where they went when nobody knew where they went. Mine it and redistribute. |
| Architecture overview | explanation | *About the architecture.* |
| Glossary | reference | |
| Security policy | reference + how-to | The policy is reference; reporting a vulnerability is a how-to. |
| Code comments | explanation, in miniature | Comments should say *why*, because the code already says *what*. |
| Error messages | reference + how-to | State what happened (reference) and what to do (how-to). Link to the fuller guide. |

## The README

A README is not a Diátaxis type. It is the front door, and it does two jobs:

1. Tell the reader **what this is** and whether it is for them. That is a short piece of
   explanation.
2. **Route them** to the four quadrants.

A good README is therefore mostly links. The failure mode is the README that grows into
a tutorial, a how-to guide, a config reference and a design rationale, all in one scroll,
because every contributor added their piece at the bottom.

```markdown
# project-name

<One paragraph: what it is, what problem it solves, who it is for.>

## Install
<The shortest path that works. Or a link.>

## Documentation
- **Tutorial**: [Your first <thing>](…), start here if you are new
- **How-to guides**: [Common tasks](…), you know what you want to do
- **Reference**: [API](…) · [Configuration](…) · [CLI](…)
- **Explanation**: [About the architecture](…) · [Why <design decision>](…)

## Contributing
<Link.>

## Licence
```

For a small project the four quadrants can be four files. The point is that they are
distinguished, not that they are large.

## Docstrings

Docstrings are reference. They describe the machinery to a reader who is at work.

- State what the function does, its parameters, its return, and what it raises.
- Do not narrate a workflow. That is a how-to guide.
- Do not justify the design. That is explanation, and it belongs in a module docstring, an
  ADR, or a comment at the decision site.
- A short usage example is good. An example that grows a story is explanation escaping.

## Which quadrant is missing?

The typical software project, in order of what it has:

1. **Reference**: usually present, often auto-generated, sometimes the only thing.
2. **How-to guides**: some, usually named after tools rather than goals.
3. **Tutorial**: one, written at launch, quietly broken since.
4. **Explanation**: almost always absent, or scattered as asides in the other three.

So the two highest-yield actions in most repositories are:

- **Re-run the tutorial from a clean environment.** It is probably broken. A broken tutorial
  is worse than no tutorial, because it fails the newcomer at the exact moment you were
  trying to win them.
- **Write one explanation page.** Gather the *why* currently scattered through the other
  docs into *About <the central concept>*. This usually improves the other three
  documents at the same time, because you are removing the material that was interrupting
  them.

## Docs for AI agents

If a repo carries agent-facing instructions (CLAUDE.md, AGENTS.md, .cursorrules), the same
analysis applies, with one adjustment: an agent is always **at work**, never at study. It
does not need a tutorial and it does not need explanation for its own sake.

So agent instructions should be **how-to guides and reference**, and explanation only where
a *why* actually changes a decision the agent will make ("we use X not Y, so do not
introduce Y").

The common failure is the same as everywhere else: a single file mixing project background,
build commands, style opinions, and architecture notes, which is hard to follow precisely
because it never says which mode the reader is in.

## Documentation you inherit

When you land in an unfamiliar repo and are asked to "improve the docs":

1. Read the README. Classify what is actually in it.
2. Try the install and the quickstart, literally, from clean. Note where they break. This
   is the highest-signal 15 minutes available anywhere in documentation work.
3. Run the audit in `references/auditing.md`.
4. Fix the broken thing you found in step 2 first. Ship it.
5. Then take the next-highest ranked item.

Do not start by proposing a docs site, a static site generator, or a directory tree. Those
are answers to a question nobody has yet asked.
