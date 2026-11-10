---
name: diataxis
description: Write, organise, audit, or fix technical documentation using the Diátaxis framework (tutorials, how-to guides, reference, explanation). Use when creating or reviewing a README, docs site, tutorial, how-to guide, API reference, or conceptual guide; when deciding where a new doc belongs; when documentation is bloated, disorganised, or "we have docs but nobody can find anything"; or when asked to restructure, split, or improve existing documentation.
license: CC-BY-SA-4.0
metadata:
  derived_from: "Diátaxis by Daniele Procida, https://diataxis.fr/ (CC BY-SA 4.0)"
  tags: documentation writing technical-writing information-architecture diataxis
---

# Diátaxis

A systematic approach to technical documentation. Documentation serves four distinct
user needs. Each need demands a different kind of document, written differently. Most
bad documentation is bad because it tries to serve two needs at once and serves neither.

Use this skill for any documentation work: writing new docs, reviewing existing ones,
deciding where something belongs, or fixing a docs tree that has gone wrong.

## Rule 0: classify before writing a word

Never start drafting until you have named the type. If you cannot name it, the document
does not yet have a purpose, and whatever you write will drift into a mixture.

If the user's request itself is a mixture ("write docs for the auth module"), split it
into separate documents by type before writing. Say what you split and why.

## The two dimensions

Documentation serves a practitioner of a craft. Craft has exactly two dimensions:

- **Action vs cognition**: practical knowledge (knowing *how*, what we do) vs theoretical
  knowledge (knowing *that*, what we think).
- **Acquisition vs application**: being at **study** (acquiring the craft) vs being at
  **work** (applying it).

Two dimensions, four quadrants. That is why there are four types and not three or five.
The map is complete: there is no other territory to cover.

```
                       ACTION  (practical steps, doing)
          ┌──────────────────────────┬──────────────────────────┐
          │  TUTORIALS               │  HOW-TO GUIDES           │
          │  learning-oriented       │  goal-oriented           │
 ACQUIS-  │  "Can you teach me…?"    │  "How do I…?"            │  APPLI-
 ITION    │  form: a lesson          │  form: a series of steps │  CATION
 (study)  ├──────────────────────────┼──────────────────────────┤  (work)
          │  EXPLANATION             │  REFERENCE               │
          │  understanding-oriented  │  information-oriented    │
          │  "Why…? Tell me about…"  │  "What is…?"             │
          │  form: discursive prose  │  form: dry description   │
          └──────────────────────────┴──────────────────────────┘
                    COGNITION  (theoretical knowledge, thinking)
```

Adjacent quadrants share a dimension and therefore blur into each other. Diagonal
quadrants have nothing in common. Tutorial/how-to is the most common and most damaging
conflation in software documentation.

## The compass

When intuition is unsure, or worse, confidently wrong, answer two questions.

| If the content…   | …and serves the user's… | …then it must be… |
| ----------------- | ----------------------- | ----------------- |
| informs action    | acquisition of skill    | a **tutorial**     |
| informs action    | application of skill    | a **how-to guide** |
| informs cognition | application of skill    | **reference**      |
| informs cognition | acquisition of skill    | **explanation**    |

Use the terms loosely, not legalistically: action = doing, cognition = thinking,
acquisition = study, application = work. Apply the compass at any scale: a whole docs
tree, one page, one paragraph, one sentence.

The decisive question for the acquisition/application axis: **would someone read this
with their hands on the machinery, mid-task, or only after stepping away from the work?**

## The four contracts

| | Tutorial | How-to guide | Reference | Explanation |
|---|---|---|---|---|
| **Serves** | study | work | work | study |
| **Reader is** | a learner, may not know what to ask | already competent, knows the goal | already competent, needs a fact | anyone, away from the product |
| **Goal** | a successful learning experience | a completed task | truth and certainty | understanding |
| **Led by** | what the learner must encounter | the user's real-world problem | the structure of the product | a topic, bounded by the author |
| **Path** | single line, no choices | forks and branches, "if x then y" | n/a, it is consulted not read | discursive, considers alternatives |
| **Responsibility** | the author's | the user's | the author's for accuracy | shared |
| **Must not** | explain, offer options, fail | teach, digress, be exhaustive | instruct, explain, opine | instruct, describe machinery |
| **Analogy** | teaching a child to cook | a recipe | the food packet label | a book on culinary history |

Read the matching reference file before writing:

| Writing… | Read |
|---|---|
| a tutorial | `references/tutorials.md` |
| a how-to guide | `references/how-to-guides.md` |
| reference material | `references/reference.md` |
| an explanation / concept guide | `references/explanation.md` |
| auditing or fixing existing docs | `references/auditing.md` |
| planning docs work, or asked for a big restructure | `references/workflow.md` |
| judging whether docs are actually good | `references/quality.md` |
| applying this to a real repo (README, docstrings, ADRs…) | `references/in-practice.md` |

## Tutorial vs how-to: the distinction that matters most

Both are practical, both are ordered steps, both promise success. They are still
completely different documents because they serve different needs.

- A **tutorial** is a lesson. Its obligation is that the learner *learns*. The author
  controls the environment, eliminates the unexpected, and carries responsibility for
  the outcome. It is concrete, single-path, safe, and repeatable.
- A **how-to guide** directs work in the real world. Its obligation is that the task
  *gets done*. It assumes competence, branches to meet real conditions, and cannot
  promise safety.

Not the difference between basic and advanced. An advanced course for a specialist is
still a tutorial. A trivial checklist for a routine chore is still a how-to guide. The
type follows the need, never the difficulty.

## Do not impose the structure

This is where documentation projects, and AI assistants in particular, go wrong.

**Never create four empty folders named tutorials/how-to/reference/explanation and call
it progress.** Diátaxis is a guide, not a plan. Structure emerges from the inside as
individual pieces are improved; it is not imposed from above. Empty scaffolding is worse
than the mess it replaced.

**Never propose a big-bang reorganisation** unless the user explicitly asks for one, and
even then, say plainly that incremental is the recommended path. The prescribed loop is:

1. **Choose** something. Whatever is in front of you. Do not go hunting for problems.
   Smaller is better: a page, a paragraph, a sentence.
2. **Assess** it. What user need does this represent? How well does it serve it? Do its
   language and logic match that mode?
3. **Decide** the single next action that produces an immediate improvement.
4. **Do it**, and ship it. One improvement is a complete unit of work.

Then repeat. Documentation is never *finished*, but it can be *complete* at every stage:
useful, coherent, and ready for the next step.

## Fast blur detection

Symptoms that a document is serving two needs at once, and the fix:

| Symptom | Diagnosis | Fix |
|---|---|---|
| "This is happening because, under the hood…" inside a step list | explanation leaking into tutorial or how-to | cut to one plain sentence, link out to the explanation |
| A tutorial offering "you can also use `--flag` for…" | options leaking into a tutorial | delete; tutorials have one path |
| A how-to guide that starts by teaching the basics | tutorial leaking into how-to | assume competence, link to the tutorial |
| A how-to guide titled after a class, tool, or screen | machinery-led, not need-led | retitle to the user's goal, or it is reference |
| Reference pages containing "first, do this, then that" | instruction leaking into reference | move steps to a how-to guide |
| Reference with a growing worked example that starts saying *why* | explanation leaking into reference | split the *why* into an explanation page |
| An explanation page that ends with installation steps | instruction leaking into explanation | move steps out, link to them |
| A doc titled "Overview", "Getting Started", "Guide", "Advanced Usage" | type-less bucket | run the compass on its contents; it is usually 2 to 4 documents |
| A tutorial that cannot be run twice, or is not reversible | reliability failure | redesign so every step can be repeated |

## Naming

Titles are the strongest signal of type, and the cheapest thing to fix.

- Tutorial: *Getting started with X* / *Your first X* / *Building a Y with X*
- How-to: *How to integrate application performance monitoring*. Not *Integrating APM*
  (maybe it argues whether you should), and never bare *Application performance
  monitoring* (unrecoverably ambiguous).
- Reference: the name of the thing. *`fetch_user()`* / *Configuration options* / *CLI flags*
- Explanation: you should be able to put an implicit *About* in front of it.
  *About user authentication*, *Why connections are pooled*. Acceptable section names:
  Discussion, Background, Concepts, Topics.

## Attribution

Diátaxis is the work of Daniele Procida, https://diataxis.fr/, CC BY-SA 4.0. This skill
is a derivative work under the same licence. It restates and condenses the framework for
use by an agent; it is not the authoritative source. Where this skill and diataxis.fr
disagree, diataxis.fr is right.
