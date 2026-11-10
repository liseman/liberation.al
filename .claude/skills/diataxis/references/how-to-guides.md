# How-to guides

**Goal-oriented. Serves the user at work. It directs action towards a result.**

A how-to guide takes the reader through a problem or towards a result. It helps them get
something done, correctly and safely. It is concerned with work: navigating from one side
of a real-world problem-field to the other.

Good scope: *how to calibrate the radar array*, *how to use fixtures in pytest*, *how to
configure reconnection back-off policies*. Bad scope: *how to build a web application*,
which is not a goal but an open-ended sphere of skill.

The list of how-to guides is also a portrait of what your product can actually do. A rich
list reads as capability. These are usually the most-read pages in a docs set.

## Addressed to problems, not to machinery

A how-to guide must be written from the perspective of the user, not the machinery. Every
guide should answer to a human project. The common failing is to define guides by
operations the tool can perform, which offers the user almost nothing.

Consider:

> "To shut off the flow of water, turn the tap clockwise."
> "To deploy the desired database configuration, select the appropriate options and press Deploy."

These look like guidance but are not. Anyone competent in the domain already knows this;
standardised interfaces and general knowledge cover it. Worse, they are disconnected from
purpose. What the user actually needs is *how much water to run, and how vigorously, for a
given purpose*, or *which configuration options match which real-world need*.

Meaning comes from purpose and need. There is no purpose in the functionality of a
machine; it is only causes and effects. So: **how-to guides are about goals, projects and
problems, not about tools.** Tools appear as incidental bit-players. A guide will often
cut across several tools, joined by what the human is trying to get done.

## What a how-to guide is not

- **Not a tutorial.** It serves work, not study. Conflating the two is the single most
  common and most damaging error in software documentation.
- **Not merely a procedure.** Real problems do not always reduce to a linear sequence.
  Steps fork and overlap; there are multiple entry and exit points. Often the user must
  apply judgement, and the guide should support that.

## Principles

**Maintain focus on the goal.** Characteristics: focused on a task or problem; assumes the
user knows what they want to achieve; action and only action; no digression, explanation
or teaching. The standing temptations are to explain, and to add reference detail for
completeness. Neither guides the user's work. If they matter, link to them.

**Address real-world complexity.** A guide useless for anything except the exact narrow
case you addressed is rarely valuable. You cannot cover every case, so stay open to the
range of possibilities in a way that lets the user adapt your guidance.

**Omit the unnecessary.** Practical usability beats completeness. Unlike a tutorial, a
how-to guide need not be end-to-end. Start and end at reasonable, meaningful points and
let the reader join it to their own work.

**Provide a set of instructions.** The guide is a contract: *if you are facing this
situation, you can work through it by taking these steps*. "Actions" includes thinking and
judgement, not just physical acts. Address how the user thinks as well as what they do.

**Describe a logical sequence.** Sometimes order is imposed by necessity (step two needs
step one). Sometimes it is subtler: two operations could go either way, but one sets up
the environment, or the user's thinking, for the other. That is reason enough to put it
first.

**Seek flow.** Ground the sequence in the patterns of the user's activity and thinking. A
workflow that makes the user switch tools repeatedly is clumsy, but look deeper than that.
What are you asking them to hold in mind, and for how long, before it resolves into
action? Are you making them jump back to earlier concerns, and is that avoidable? Action
has pace and rhythm; misjudged pace damages flow. At its best, a how-to guide anticipates
the user, like a helper who puts the tool in your hand as you reach for it.

**Pay attention to naming.** Titles should say exactly what the guide shows.

- good: *How to integrate application performance monitoring*
- bad: *Integrating application performance monitoring* (maybe it argues whether you should)
- very bad: *Application performance monitoring* (maybe it is how, maybe whether, maybe just what it is)

Search engines value good titles as much as humans do.

## Language

| Pattern | Purpose |
|---|---|
| *This guide shows you how to…* | state the problem or task clearly |
| *If you want x, do y. To achieve w, do z.* | conditional imperatives, the native grammar of how-to |
| *Refer to the x reference guide for a full list of options.* | do not pollute the guide with everything related to x |

## Skeleton

```markdown
# How to <achieve the user's goal>

<One or two sentences: the situation this addresses, and what you will have at the end.>

## Before you begin
<Only what blocks the task. Assume competence.>

## <Step, named for the sub-goal, not the command>
<Instruction.>

If <condition>, <alternative instruction>.

## <Next step>
...

## Verify
<How the user knows it worked.>

## Related
- <the reference page for the full option list>
- <the explanation page for the why>
```

## Checklist

- [ ] Is the title of the form *How to <do the thing the user wants>*?
- [ ] Is it named for a human goal rather than for a tool, class, or screen?
- [ ] Does it assume competence instead of teaching the basics?
- [ ] Is there zero explanation of *why*, beyond a clause and a link?
- [ ] Does it branch where the real world branches, with "if x, then y"?
- [ ] Is it adaptable, or does it only work for the one narrow case you had in mind?
- [ ] Does it stop, rather than trying to be exhaustive?
- [ ] Do option lists link out to reference rather than being reproduced inline?
- [ ] Does the order match how a person actually thinks through the task?
- [ ] Is there a way for the user to confirm success?

## Failure modes

- **Machinery-led guides.** A guide per button or per API method. If the title names a
  tool rather than a goal, it is probably reference material wearing a how-to costume.
- **Teaching creep.** Explaining what a container is, in a guide about deploying one.
- **Completeness creep.** Reproducing every flag inline. Link to reference instead.
- **Single-path rigidity.** No acknowledgement that the reader's environment differs.
- **Stating the obvious.** "Click Save to save." Domain-competent readers know this, and
  it dilutes the guidance that actually matters.
- **The open-ended non-goal.** *How to build a web application* is not a how-to guide. It
  is a curriculum, or a lie.
