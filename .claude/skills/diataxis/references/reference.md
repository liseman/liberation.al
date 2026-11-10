# Reference

**Information-oriented. Serves the user at work. It describes the machinery.**

Reference guides are technical descriptions of the machinery and how to operate it. They
contain propositional knowledge the user looks to while working. The only purpose is to
describe, as succinctly as possible and in an orderly way.

Unlike tutorials and how-to guides, which are led by the needs of the user, **reference is
led by the product it describes.**

Users need reference because they need truth and certainty: firm platforms to stand on
while they work. Reference is like a map. It tells you what you need to know about the
territory without going out and checking the territory yourself.

One hardly *reads* reference material. One *consults* it.

## Principles

**Describe and only describe.** Neutral description is the key imperative.

- Style: austere and uncompromising
- Values: neutrality, objectivity, factuality
- Structure: according to the structure of the machinery itself

Neutral description is one of the hardest things to do, because it is not a natural way of
communicating. What comes naturally is to explain, instruct, discuss and opine, and all of
those run counter to what reference demands: accuracy, precision, completeness, clarity.
The temptation is strong precisely because description can feel too thin to be useful, and
because those other things are needed. They are needed *elsewhere*. Link to them.

Reference may still state how something works, or the correct way to use it. Stating a
constraint is description. Walking the user through a task is not.

**Adopt standard patterns.** Reference is useful when it is consistent. Standard patterns
are what make it usable at speed. Put the material where the user expects it, in a format
they recognise. Writing offers many opportunities to delight readers with vocabulary and
range; reference is emphatically not one of them. Every entry of the same kind should have
the same shape.

**Respect the structure of the machinery.** A map is usable because it corresponds to the
territory. The structure of the documentation should mirror the structure of the product,
so the user can work through both at once. This does not mean forcing an unnatural
hierarchy; it means letting the logical and conceptual arrangement of the code make sense
of the docs.

**Provide examples.** Examples illustrate without derailing description. A short usage
example for a command conveys its shape and context economically. Keep them illustrative.
The moment an example starts saying *why*, or *what if*, or *how it came to be*, it has
become explanation and belongs elsewhere.

**Auto-generation is a tool, not a strategy.** Generating API reference from source keeps
it faithful to the code, which is valuable. It is not "the documentation". It covers one
quadrant only, and too many software projects stop there.

## Language

| Pattern | Purpose |
|---|---|
| *Django's default logging configuration inherits Python's defaults. It is available as `django.utils.log.DEFAULT_LOGGING` and defined in `django/utils/log.py`.* | state facts about the machinery and its behaviour |
| *Sub-commands are: a, b, c, d, e, f.* | list commands, options, operations, features, flags, limitations, error messages |
| *You must use a. You must not apply b unless c. Never d.* | provide warnings where appropriate |

Imperatives appear in reference only as statements of constraint ("you must not"), never
as steps in a procedure.

## The food packet test

You check the label on a packet of food to make a decision. You do not want opinions,
speculation, instructions or interpretation. You expect a standard presentation, so you
can find nutritional values, storage, ingredients and allergens quickly, and rely on them.

*May contain traces of wheat.* *Net weight: 1000g.*

You would not expect recipes or marketing claims mixed in with that, and if they were, it
could be literally dangerous. Food labelling is governed by law for exactly this reason.
The same seriousness should apply to technical reference.

## Skeleton

```markdown
# <name of the thing>

<One-sentence description of what it is.>

## Signature / Syntax
<Exact form.>

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| … | … | … | … |

## Returns
<Type and meaning.>

## Raises / Errors

| Condition | Result |
|---|---|
| … | … |

## Constraints
- You must …
- You must not … unless …

## Example
<Minimal, illustrative, uncommented beyond the essential.>

## See also
- <how-to guide that uses this>
- <explanation of why it behaves this way>
```

## Checklist

- [ ] Is every sentence a statement of fact about the machinery?
- [ ] Is there any instruction ("first do this, then that")? Remove it.
- [ ] Is there any justification or history ("we chose this because")? Remove it.
- [ ] Is there any marketing or evaluation ("powerful", "simple", "blazing fast")? Remove it.
- [ ] Do all entries of the same kind follow an identical template?
- [ ] Does the structure mirror the product's own structure?
- [ ] Is it complete for the surface it claims to cover?
- [ ] Are examples short and purely illustrative, with no "why"?
- [ ] Could a user find a given fact in a few seconds without reading prose?
- [ ] Is it accurate against the current version, and is there a mechanism to keep it so?

## Failure modes

- **Narrative reference.** Prose paragraphs where a table belongs.
- **Tutorial creep.** A "getting started" section growing inside the API reference.
- **Explanation creep.** The design rationale of a parameter attached to that parameter.
  Split it out and link.
- **Inconsistent entries.** Three functions documented three different ways. This is the
  quiet killer, because it destroys the scanning speed that reference exists for.
- **Editorialising.** "This is the recommended approach" is explanation, or a how-to guide.
- **Generated-and-abandoned.** Docstrings emitted as HTML with no attention paid to
  whether the result is complete, ordered, or navigable.
