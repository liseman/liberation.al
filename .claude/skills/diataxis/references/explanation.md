# Explanation

**Understanding-oriented. Serves the user at study. It is discussion, not instruction.**

Explanation is a discursive treatment of a subject that permits reflection. It deepens and
broadens understanding, bringing clarity, light and context.

Reflection comes *after* something and depends on it, yet brings something new: it shines
a fresh light on matter already encountered. So explanation takes a higher and wider
perspective than the other three types. It does not take the user's eye-level view like a
how-to guide, or the close-up view of the machinery like reference. Its scope is a
**topic**: an area of knowledge, bounded in some reasonable, meaningful way.

For the user, explanation joins things together. It answers *"Can you tell me about…?"*

It is the only kind of documentation it makes sense to read away from the product. One
could say it is the only kind you might read in the bath.

## Why it matters

Explanation is characterised by its distance from the practitioner's active concerns. It
has no direct implication for what they do right now, so it is often treated as less
important. That is a mistake. It is less **urgent**, not less **important**, and it is not
a luxury. No practitioner of a craft can afford to be without an understanding of that
craft.

Understanding does not come *from* explanation, but explanation is required to form the
web that holds everything together. Without it, a practitioner's knowledge is loose,
fragmented and fragile, and their exercise of the craft is **anxious**.

The etymology is instructive. *Explanation* and its cognates mean unfolding, revealing
what is hidden in the folds. *Understanding* shares roots with holding and grasping, as in
*comprehend*. Understanding seals together the other components of mastery and makes the
craft safely our own.

## The hardest quadrant

Tutorials, how-to guides and reference are all bounded by something external and
well-defined: what the user must learn, what task they must achieve, or the extent of the
machinery itself. Explanation has no such anchor. Where does one start? Where does one
conclude? The open-endedness offers the writer too many possibilities.

Two remedies:

1. Use a real or imagined **"why?" question** as the prompt, and answer that.
2. Failing that, draw lines that mark out a reasonable area, and be satisfied with them.

Because explanation is rarely recognised explicitly, it tends to be scattered in small
parcels through the other sections instead of being written. Collecting those parcels into
a real explanation page is usually the single highest-value improvement available in a
mature docs set.

## Principles

**Make connections.** You are weaving a web of understanding. Connect to other things,
even outside the immediate topic, if it helps.

**Provide context.** Explain why things are as they are: design decisions, historical
reasons, technical constraints. Draw implications. Mention specific examples.

**Talk about the subject.** Explanation guides are about a topic in the sense of being
*around* it. Discuss:

- the bigger picture
- history
- choices, alternatives, possibilities
- why: reasons and justifications

The titles should reflect this. You should be able to put an implicit, or even explicit,
*About* in front of each one: *About user authentication*, *About database connection
policies*.

**Admit opinion and perspective.** Opinion seems a strange thing to introduce into
documentation, but all human activity and knowledge is invested with beliefs and thoughts,
and the reality of any human creation is rich with opinion. Any understanding also comes
from a standpoint, which means other standpoints exist. Explanation can and must consider
alternatives, counter-examples, and different approaches to the same question. Think of it
as discussion: discussions can weigh contrary opinions.

**Keep it closely bounded.** Explanation tends to absorb other things. The writer, intent
on covering the topic, feels the urge to include instruction or technical description.
Documentation already has places for those. Letting them creep in damages the explanation
and removes that material from where it belongs.

## Naming

Explanation sections need not be called *Explanation*. Alternatives that read naturally:

- Discussion
- Background
- Conceptual guides
- Topics

## Language

| Pattern | Purpose |
|---|---|
| *The reason for x is that historically, y…* | explain |
| *W is better than z, because…* | offer judgements, and opinions where appropriate |
| *An x in system y is analogous to a w in system z. However…* | provide context that helps the reader |
| *Some users prefer w (because z). This can be a good approach, but…* | weigh alternatives |
| *An x interacts with a y as follows:…* | unfold the internal workings, to help understand why something behaves as it does |

## Skeleton

```markdown
# About <topic>

<The question this page answers, stated plainly. Often a "why".>

## <The situation, or the problem the design responds to>
<Context. What was true, what was needed.>

## <How it works, at the level of ideas>
<Mechanism as understanding, not as reference. No signatures, no step lists.>

## Why it is this way
<Design decisions. Historical reasons. Technical constraints. Trade-offs accepted.>

## Alternatives, and why not
<Other approaches. What they are good for. Where they were rejected and why.>

## Implications
<What this means for how you work with it.>

## See also
- <the how-to guide for doing it>
- <the reference for the exact surface>
```

## Checklist

- [ ] Can you put *About* in front of the title and have it read correctly?
- [ ] Does it answer a "why" or a "tell me about" question?
- [ ] Does it contain instructions? Move them out.
- [ ] Does it contain parameter tables or signatures? Move them to reference.
- [ ] Does it make connections beyond the immediate topic?
- [ ] Does it give historical or design context, not just mechanism?
- [ ] Does it consider at least one alternative or counter-position?
- [ ] Is it bounded, or has it sprawled into being a second manual?
- [ ] Would it still make sense read away from a keyboard?

## The McGee test

Harold McGee's *On Food and Cooking* (1984) teaches you to cook nothing. It contains no
recipes except as historical examples, and it is not a work of reference. It places food
and cooking in the context of history, society, science and technology, and explains why
we do what we do in the kitchen and how that changed.

It is clearly not a book you read *while* cooking. You read it when you want to reflect on
cooking. It illuminates the subject by shining light from several angles.

Afterwards your understanding is changed. What you learned may not be applicable next time
you are at the stove, but it will change how you think about your craft, and that will
affect your practice.

If your explanation page would be equally at home open next to a terminal mid-task, it is
probably reference or a how-to guide.

## Failure modes

- **The everything page.** Explanation absorbs setup steps, config tables and a tutorial,
  and becomes the de facto manual.
- **The un-written quadrant.** No explanation exists at all; the *why* is scattered as
  asides through tutorials and reference, where it obstructs both.
- **Mechanism without meaning.** Restating what reference already says, in longer
  sentences, and calling it a concept guide.
- **False neutrality.** Refusing to say what the trade-offs were, or which approach the
  project prefers and why. Explanation is where opinion is legitimate.
- **Unbounded sprawl.** No "why" question driving it, so it never finds an ending.
