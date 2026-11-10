"""What a test asks, as the structured query the answer code takes."""

from typing import Any

from liberational.memory import history_qa, question


def query(asked: str, state: Any) -> dict[str, Any]:
    """The query for a question, without its window: its fixed reading (the `readings` fixture) and what the names
    it picked are in `state`."""
    return history_qa.describe(asked, question.read(asked), state)
