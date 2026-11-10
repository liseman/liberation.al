"""The domain model: plain, immutable types for people, identity assertions, encounters, and events.

These are steps 1 and 2 of docs/DOMAIN-MODEL.md. Step 3 (`liberational.people.registry`) writes people and identity
assertions in these shapes; later steps make the other stores write them too.

- `kernel`: what every context shares (ids, the instant, evidence references, producers, the event envelope and
  catalog). It is the only module the contexts import from each other.
- `identity`, `memory`: one module per bounded context (people and their assertions; encounters).
- `eventlog`: the event log port (step 2). Its adapter, which does the I/O, is `liberational.knowledge_eventlog`.

No module here does input or output, or imports a model, camera, or database library. Every type is a frozen
dataclass that checks its own invariants when it is built, so an invalid record fails where it is made.
"""
