# Auditing and fixing existing documentation

Use this when asked to review, restructure, split, or "clean up" documentation that
already exists.

The governing constraint from `references/workflow.md` still applies: **do not propose a
big-bang reorganisation.** An audit produces a ranked list of small, independently
shippable improvements, not a migration plan.

## The audit procedure

**1. Inventory.** List the documents. For each, record its title, its apparent purpose,
and its length. Do not read everything in depth yet.

**2. Classify.** Run the compass on each document and assign one of: tutorial, how-to,
reference, explanation, or **mixed**. Record the evidence, usually a title or an opening
sentence.

**3. Find the gaps.** Which quadrants are empty or thin? The near-universal pattern in
software projects is plenty of reference (often auto-generated), some how-to guides, one
weak tutorial, and no explanation at all.

**4. Find the blur.** For every document marked mixed, note where the type changes. The
transition is usually visible at a specific paragraph.

**5. Check titles.** Titles are the cheapest fix with the highest yield. A wrong title
sends the wrong reader to the wrong document, which is worse than the document being
imperfect.

**6. Rank by cost and benefit.** Prefer changes that are small, safe, and immediately
shippable. Deleting a misleading paragraph today beats a perfect restructure next quarter.

**7. Report and proceed.** Give the user the ranked list. Then start at the top, one item
at a time, shipping each.

## Classification evidence

Quick signals, in rough order of reliability:

| Signal | Points to |
|---|---|
| Numbered steps that must be followed in order, from a fixed starting state | tutorial or how-to |
| Steps with "if your setup is X, do Y instead" | how-to |
| Steps that assume nothing and control the environment | tutorial |
| Tables of parameters, flags, fields, error codes | reference |
| Prose that says "because", "historically", "the trade-off" | explanation |
| Bored on reading it, but you would return to look something up | reference |
| You could read it away from the machine | explanation |
| The reader must already know what they want | how-to or reference |
| The reader may not yet know what to ask | tutorial or explanation |

When two signals conflict, the document is mixed. That is the finding, not a problem with
the method.

## Symptom to remedy

| Symptom | Diagnosis | Remedy |
|---|---|---|
| Long "Getting Started" that installs, explains architecture, and builds an app | all four types in one page | split into: install (how-to), first app (tutorial), About the architecture (explanation) |
| Tutorial with "Note: you could also…" boxes | option creep | delete; a tutorial has one path |
| Tutorial with a paragraph of theory between steps | explanation creep | reduce to one sentence and link out |
| How-to guide that starts with "First, some background" | explanation creep | move the background to an explanation page, link to it |
| How-to guide reproducing the full flag list | reference creep | link to reference, keep only the flags this task uses |
| How-to guide named after a class, endpoint, or UI screen | machinery-led | retitle to the goal, or reclassify as reference |
| Reference entry with a rationale paragraph | explanation creep | extract to *About <topic>*, leave a link |
| Reference entry with a walkthrough | how-to creep | extract to *How to <goal>* |
| Explanation page ending in setup instructions | how-to creep | extract; explanation does not instruct |
| Two pages covering the same thing for different readers | not duplication | usually correct: a tutorial and a how-to guide legitimately overlap in subject |
| A page nobody can classify | it has no user need | either find the need and rewrite, or delete it |
| Four empty directories from a previous "Diátaxis migration" | scaffolding imposed from outside | fill or remove; do not leave empty structure standing |

## Splitting a mixed document

The mechanical recipe, applied to one document at a time:

1. **Mark the seams.** Go through and label each block with its type. The seams are
   usually obvious once you look for them.
2. **Pick the dominant type.** Whatever most of the document is, and whatever its title
   promises, that is what stays at this URL.
3. **Cut the foreign material out**, into a new document of the correct type. Give it a
   title following the naming rules for that type.
4. **Replace the cut with a link**, at most a sentence long. Not a summary; a link.
5. **Reread the remainder** as the reader of the dominant type. It usually reads better
   and shorter than it did before.
6. **Ship it.** Do not wait to split the next document.

Keep the original URL working. Breaking links is a real cost paid by real users, and it is
rarely justified by a tidier tree.

## What not to do

- **Do not create empty quadrant folders.** Structure emerges from improved content, not
  the reverse.
- **Do not rename everything at once** to fit a naming convention. Titles matter, but a
  hundred broken links matter more.
- **Do not delete explanation you cannot place.** Park it in an *About* page; it is usually
  the scarcest material in the set.
- **Do not treat overlap as duplication.** The same subject legitimately appears in all
  four quadrants. *Authentication* can be a tutorial, a how-to guide, a reference page and
  an explanation, all at once, all correct.
- **Do not report only structural findings.** Diátaxis exposes lapses in accuracy and
  completeness, but it does not fix them. If you find something factually wrong, that is
  the more urgent finding.

## Reporting format

```markdown
## Inventory
| Document | Current type | Should be | Note |
|---|---|---|---|

## Gaps
- <quadrant>: <what is missing and who it fails>

## Ranked improvements
1. <single next action>, <file>, <why it wins>
2. …
```

Keep each ranked item to a single next action that produces an immediate improvement. If
an item cannot be shipped on its own, it is too big; break it down.
