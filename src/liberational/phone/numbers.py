"""Phone numbers in E.164 form (+ and a country code), for comparing an owner's number with a caller's.

Kept free of other imports so config.save() can check owner.phone with it.
"""

import re

_FORMATTING = re.compile(r"[\s().-]")
_E164 = re.compile(r"\+[1-9][0-9]{7,14}")  # ASCII digits only: Python's \d also takes other scripts' digits
# "+44 (0)20 ..." writes the national trunk 0 beside the country code. Dropping the parentheses would keep a 0 the
# international number doesn't have, and in some countries (Italy) the 0 is part of it, so it's refused, not fixed.
_TRUNK_ZERO = re.compile(r"\(\s*0\s*\)")


def e164(number: str | None) -> str | None:
    """number as +<country code><number>, or None when it can't be read that way for certain.

    Spaces, dashes, dots, and parentheses are dropped, so "+1 (555) 555-0123" is "+15555550123". A number must
    start with + and have 8 to 15 digits after it, the first not 0. One with no +, like "555-555-0123", is None: its
    country can't be known, and guessing could make a different number match. So is one with a trunk "(0)"."""
    if _TRUNK_ZERO.search(number or ""):
        return None
    text = _FORMATTING.sub("", (number or "").strip())
    return text if _E164.fullmatch(text) else None
