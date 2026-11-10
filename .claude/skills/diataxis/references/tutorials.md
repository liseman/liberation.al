# Tutorials

**Learning-oriented. Serves the user at study. It is a lesson, not a task.**

A tutorial is an experience under the guidance of a tutor. The user learns by doing
something meaningful towards an achievable goal. Its purpose is not to help the user get
something done. Its purpose is to help them learn.

What the learner *does* is not necessarily what they *learn*. A tutorial framed as
"build a to-do app" may actually be teaching the shape of the request cycle, the names of
things, which tool does what, and the rhythm of the workflow. Know both: the surface task
and the real curriculum.

## The contract

Nearly all responsibility falls on the author. The learner's only obligation is to be
attentive and follow directions. They are not obliged to learn, understand, or remember.

The exercise you put them through must be:

- **meaningful**: they need a sense of achievement
- **successful**: they must be able to complete it
- **logical**: the path through it must make sense
- **usefully complete**: they must encounter every action, concept and tool they need to
  become familiar with

## Why tutorials are hard

- A real teacher watches, corrects, and rescues. You cannot. **You are required to be
  present, but condemned to be absent.**
- The end-to-end journey must hold together, so a small product change can cascade
  through the whole narrative. No other doc type is revised like this.
- You must separate what is to be *learned* from what is to be *done*, and then invent a
  journey that delivers the former through the latter.

Budget for this. A tutorial is the most expensive document you will maintain. Two
excellent tutorials beat nine neglected ones.

## Anti-pedagogical temptations

Resist all four: **abstraction and generalisation, explanation, choices, information.**

## Principles

**Do not try to teach.** The first rule of teaching is: don't. Your job is to provide an
experience through which learning happens. The urge to impart understanding by telling is
natural and it wrecks the lesson. Give the learner things to *do*, and trust it.

**Show them where they are going.** State the destination up front so they can see
themselves building towards it: *"In this tutorial we will create and deploy a scalable
web application. Along the way we will encounter containerisation tools and services."*
Never *"In this tutorial you will learn…"*, which is presumptuous and a poor pattern.

**Deliver visible results early and often.** Understanding comes from connecting cause to
effect, so let them make that connection immediately and repeatedly. **Every step should
produce a comprehensible result, however small.** A step with no observable outcome is a
step the learner cannot verify, and verification is the whole mechanism.

**Maintain a narrative of the expected.** At every step the learner silently asks "did
that work?". Answer before they ask. Show real output, ideally exact output. Warn about
surprises: *"The command will probably return several hundred lines of logs."* Flag known
failure signs: *"If the output doesn't show X, you have probably forgotten to Y."*

**Point out what they should notice.** Learners are too busy doing to observe. Close the
loops for them in passing: how the prompt changes, what appeared in the sidebar, that the
file now exists. Observing is an active skill, and it is usually neglected.

**Target the feeling of doing.** Skill becomes a confident rhythm, and eventually a
pleasure. Tie purpose to action so the tasks cradle that feeling. Pay attention to what
the operation actually feels like to perform, and design for that.

**Encourage and permit repetition.** Learners re-run steps that worked, for the pleasure
of the expected result and the proof that they can do it. **Repetition is not the best
teacher; sometimes it is the only teacher.** Make steps repeatable and reversible
wherever you can. Avoid one-way doors: destructive setup, non-idempotent commands,
anything that punishes a second attempt.

**Ruthlessly minimise explanation.** A tutorial is not the place for it. The learner is
concentrating on doing. Explanation distracts and blocks learning in the moment. *"We're
using HTTPS because it's more secure"* is enough; link to the real discussion. Explanation
is pertinent only when the *user* wants it, and that is not the author's call. This is the
hardest temptation to resist, because once we have grasped something we naturally want to
frame it abstractly, and abstraction is the logical form of understanding but not the
route to it.

**Focus on the concrete.** *This* problem, *this* action, *this* result. Keeping to the
concrete does not deny the learner the general pattern. Minds extract patterns from
examples spectacularly well. All learning moves from the concrete and particular towards
the general and abstract, never the reverse.

**Ignore options and alternatives.** Other flags, other approaches, other ways to call the
API: ignore them. Stay on exactly what is required to reach the conclusion. This makes the
tutorial shorter and saves both of you the cognitive work.

**Aspire to perfect reliability.** Confidence is built layer by layer and easily shaken. A
learner who follows your directions and does not get the promised result loses faith in
the tutorial, in you, and in themselves. Your tutorial should be so well constructed that
things *cannot* go wrong, for every user, every time. You will not find the flaws
yourself. Find them by watching real users work through it.

## Language

| Pattern | Purpose |
|---|---|
| *We…* (first person plural) | affirms the relationship: you are not alone, we are in this together |
| *In this tutorial, we will…* | describe what the learner will accomplish |
| *First, do x. Now, do y. Now that you have done y, do z.* | no room for ambiguity or doubt |
| *We must always do x before y because… (see [Explanation])* | minimal explanation, most basic language, link out |
| *The output should look something like…* | set clear expectations |
| *Notice that… / Remember that… / Let's check…* | clues that they are on track |
| *You have built a secure three-layer stasis engine…* | describe, and mildly admire, what they achieved |

## Skeleton

```markdown
# Building your first <thing> with <product>

In this tutorial we will <destination>. Along the way we will meet <tools/concepts>.
It takes about <n> minutes.

## Before we start
<Exact prerequisites. Pinned versions. A single copy-pasteable setup.
No choices. No "or, if you prefer".>

## Step 1: <concrete action>
<One instruction.>
<Exact expected output.>
Notice that <observation the learner would otherwise miss>.

## Step 2: <concrete action>
...

## What we built
<Describe the finished thing. Admire it, mildly.>

## Next steps
- <link to the how-to guides for real work>
- <link to the explanation for the why>
```

## Checklist

Reject the draft if any answer is no.

- [ ] Does it state the destination in the first paragraph, without "you will learn"?
- [ ] Does every single step produce an output the learner can see and check?
- [ ] Is the expected output shown for each meaningful step?
- [ ] Is there exactly one path, with zero optional flags, alternatives or asides?
- [ ] Is every explanation either one plain sentence or a link?
- [ ] Can the whole thing be run a second time from a clean state without manual cleanup?
- [ ] Can a learner recover from a mistake without starting over?
- [ ] Have you actually run it end to end, from a clean environment, exactly as written?
- [ ] Are versions pinned, so it still works next month?
- [ ] Is every prerequisite stated, and does the setup work before step 1?

## Failure modes

- **It is really a how-to guide.** Ask: does the reader already know what they want to
  achieve? If yes, it is a how-to guide.
- **Explanation creep.** Paragraphs of "how this works under the hood" between steps.
- **Choice creep.** "You can use npm or yarn or pnpm." Pick one. The learner cannot
  evaluate the trade-off and should not be asked to.
- **Invisible steps.** Three commands in a row with nothing to observe.
- **Untested drift.** The commonest reason tutorials fail is that nobody re-ran them.
- **Environment optimism.** "Assuming you have Docker configured" is where the learner
  gets stranded and never returns.
