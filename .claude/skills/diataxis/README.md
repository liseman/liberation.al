# diataxis-skill

An agent skill that teaches Claude Code (and other assistants that read `SKILL.md`) to
write, organise and repair technical documentation using
[Diátaxis](https://diataxis.fr/), Daniele Procida's systematic approach to documentation
authoring.

Diátaxis identifies four distinct user needs and four corresponding forms of
documentation: **tutorials**, **how-to guides**, **reference** and **explanation**. Most
bad documentation is bad because a single page tries to serve two of those needs at once,
and serves neither.

## Why a skill

Documentation is one of the things coding agents are asked for most and get wrong most
predictably. Left alone, an assistant will happily produce a "Getting Started" page that
installs the software, teaches the concepts, walks through a build, lists every config
option and justifies the architecture, all in one scroll. That page is unusable by anyone,
for any purpose.

This skill supplies the missing discipline:

- **Classify before writing.** The compass turns "what kind of document is this?" into two
  yes/no questions with a determinate answer.
- **Per-type contracts.** What each type must do, must never do, how it should sound, and a
  checklist that produces a verdict.
- **Blur detection.** Symptom to diagnosis to remedy, for the specific ways the four types
  bleed into each other.
- **Process guardrails.** Diátaxis explicitly warns against imposing structure from above.
  Agents love imposing structure from above. The skill blocks the empty-four-folders
  reflex and enforces the one-small-improvement-at-a-time loop instead.

## Contents

```
SKILL.md                        entry point: the map, the compass, the four contracts,
                                blur detection, naming, and the workflow guardrail
references/
  tutorials.md                  learning-oriented: principles, language, skeleton, checklist
  how-to-guides.md              goal-oriented: same
  reference.md                  information-oriented: same
  explanation.md                understanding-oriented: same
  auditing.md                   procedure for reviewing and splitting existing docs
  workflow.md                   the iterative working method, and what not to do
  quality.md                    functional vs deep quality, used as a review lens
  in-practice.md                applying it to READMEs, docstrings, ADRs, runbooks,
                                changelogs, agent instructions (an extension, not Diátaxis)
```

`SKILL.md` is the only file loaded up front. It routes to the reference files as needed.

## Install

For Claude Code, as a personal skill:

```bash
git clone https://github.com/Roritharr/diataxis-skill ~/.claude/skills/diataxis
```

Or for a single project:

```bash
git clone https://github.com/Roritharr/diataxis-skill .claude/skills/diataxis
```

Then invoke it with `/diataxis`, or just ask for documentation work and let the
description trigger it.

The skill is plain Markdown with no dependencies, tools or API calls, so it works with any
assistant that reads the `SKILL.md` convention.

## Using it

It triggers on documentation work generally. Things it handles well:

- *"Write a how-to guide for configuring retries."*
- *"Where should this doc live?"*
- *"Our docs are a mess, audit them."* → inventory, classification, gaps, ranked fixes
- *"Split this README."*
- *"Review this page."* → against the type's contract, then the quality lens

## Attribution and licence

Diátaxis is the work of **Daniele Procida** and is published at
[diataxis.fr](https://diataxis.fr/) under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

This repository is a **derivative work** of that material and is licensed under the same
terms: [CC BY-SA 4.0](LICENSE).

**Modifications made:** the framework has been condensed and restructured into agent skill
format. The concepts, the map, the compass and the four types are Procida's, and the
principles for each type are restatements and paraphrases of the source. Added here are:
the per-type checklists and document skeletons; the symptom/diagnosis/remedy tables; the
audit and splitting procedures; the agent-specific process guardrails; and
`references/in-practice.md`, which applies the framework to repository artefacts and is not
part of Diátaxis.

This is not the authoritative source. Where this skill and diataxis.fr disagree,
diataxis.fr is right. Read the original. It is shorter than this, and better written.
