"""Tags with attributes: "that's Alice, CEO of Acme" as a name, a role, and an organization (step 6 of
docs/DOMAIN-MODEL.md).

A tag decides who someone is, so it is parsed when it is written, by fixed rules and with no model call (the
identity path is schema-on-write). The same parser reads a spoken tag (the identity daemon and the assistant's
"that's Alice" command, through `people/identity.py`) and a typed one (the dashboard).

A sentence parses only when all of it is one person's name and, optionally, one attribute clause. Anything else
gives no tag, so a sentence the rules do not cover costs a missing role, not a wrong one:

    that's Alice, CEO of Acme              Alice, role CEO, organization Acme
    this is Dana, the head of sales at Initech
    meet Dana, our new CTO                 role only
    I'm Alice, I work at Acme              organization only
    that's Alice, Acme's CEO
    that's Alice, she's here               no tag: "here" is not a role
    that's Alice and Dana                  no tag: two people

The rules lean to precision:

- A role ends in a word from `ROLE_HEADS` ("CEO", "engineer", "teacher"). Words for a relationship ("friend",
  "boss", "partner" without an organization) are not roles.
- An organization is a run of capitalized words. It follows a role after "of" or "at", or follows "works at". "From"
  and "with" are left out, because "Alice from Boston" and "Alice with Dana" name a place and a person, not an
  employer. For the same reason "for" and the possessive ("Acme's CEO") take only a role that is held at an
  organization (`ORG_HEADS`): "Dana's lawyer" and "a lawyer for Bob Smith" name a client, and "works for" is left out.
- "I'm" and "I work" describe the speaker, so they count only after the speaker's own name ("I'm Alice, I work at
  Acme"). After "that's Alice" they say nothing about Alice.
- Without a comma, a role needs an article ("that's Alice the CEO of Acme"), so a surname that is also a job ("Mary
  Cook") stays part of the name.
"""

import dataclasses
import re
from collections.abc import Callable

# Lead-ins. The owner's command forms match people/identity.INTRO_COMMAND; the rest are introductions in conversation.
COMMAND_LEADS = r"that's|that is|this is|meet|say hi to"
THIRD_LEADS = COMMAND_LEADS + r"|say hello to|here's|here is|introducing|introduce you to|this here is"
SELF_LEADS = r"I'm|I am|my name is|my name's|the name's|call me"
SPOKEN_LEADS = THIRD_LEADS + "|" + SELF_LEADS
# "this is my friend Dana": the relationship to the speaker is not the person's role, so it is skipped.
_POSSESSIVE = r"(?:my\s+(?:[a-z]+\s+){0,2}(?=[A-Z]))?"

ROLE_HEADS = frozenset(
    """
    ceo cto cfo coo cmo cio ciso cpo cro vp svp evp president founder cofounder co-founder chair chairman chairwoman
    chairperson director manager engineer developer programmer designer scientist researcher professor lecturer
    teacher student intern analyst consultant lawyer attorney doctor physician surgeon nurse dentist therapist
    pharmacist architect accountant lead head officer editor writer reporter journalist producer investor advisor
    adviser recruiter coordinator administrator assistant secretary treasurer chef cook coach trainer pastor priest
    rabbi imam agent representative rep salesperson contractor plumber electrician mechanic pilot artist musician
    photographer specialist technician supervisor executive principal dean chancellor provost mayor senator governor
    ambassador librarian paramedic firefighter veterinarian vet postdoc fellow
    """.split()  # noqa: SIM905 (word lists read better as text)
)
# Roles held at an organization, never for a person, so "for ORG" and "ORG's ROLE" can name the organization.
ORG_HEADS = frozenset(
    """
    ceo cto cfo coo cmo cio ciso cpo cro vp svp evp president founder cofounder co-founder chair chairman chairwoman
    chairperson director officer
    """.split()  # noqa: SIM905 (word lists read better as text)
)
# Heads that also name a relationship ("my partner") or a bare membership: a role only when an organization is named.
ORG_ONLY_HEADS = frozenset({"partner", "associate", "member", "owner", "employee"})
# "VP of Engineering" is one role; "CEO of Acme" is a role and an organization.
DEPARTMENTS = frozenset(
    """
    sales marketing engineering product products design operations finance hr people research technology
    communications partnerships growth data security legal strategy development business customer success support
    admissions students infrastructure platform it analytics policy content
    """.split()  # noqa: SIM905 (word lists read better as text)
)
_DETERMINERS = frozenset({"a", "an", "the", "our", "my", "your", "their", "his", "her"})
# Words that never start or pad a role, so "she's here with the engineer" is not a role.
_FUNCTION_WORDS = _DETERMINERS | frozenset(
    """
    of at for from with in on to by and or but not no here there now then just also so very really is was be been
    she he they who i we you it this that what this these those me him them us
    """.split()  # noqa: SIM905 (word lists read better as text)
)
_ROLE_PREPOSITIONS = ("of", "at", "for")
_ORG_WORD = re.compile(r"[A-Z0-9][\w&.'+-]*")
_ORG_JOINERS = frozenset({"of", "&", "de"})


def _clause(self_intro: bool) -> re.Pattern[str]:
    """What may follow a name. The first person ("I'm", "I work") only after the speaker's own name."""
    be, who, verb = "she's|he's|they're|who's|she is|he is|they are|who is", "she|he|they|who", "works|work|worked"
    if self_intro:
        be, who, verb = be + "|I'm|I am", who + "|I", verb + "|am working"
    be, works = rf"(?:{be})", rf"(?:(?:{who})\s+)?(?:{verb}|is working|are working)"
    return re.compile(
        rf"(?P<and>and\s+(?=(?:{be}|{works})\b))?"
        rf"(?:(?P<works>{works})\s+at\s+(?P<org>.+)|(?:(?P<be>{be})\s+)?(?P<rest>.+))",
        re.IGNORECASE,
    )


_CLAUSES = {self_intro: _clause(self_intro) for self_intro in (False, True)}
MAX_NAME_WORDS = 4


@dataclasses.dataclass(frozen=True, slots=True)
class Tag:
    """One person's name, and the role and organization said with it, if any."""

    name: str
    role: str | None = None
    organization: str | None = None

    def attributes(self) -> dict[str, str]:
        """The attributes that were said, for `Registry.describe`."""
        return {k: v for k, v in (("role", self.role), ("organization", self.organization)) if v}

    def described(self) -> str:
        """The attributes as a short phrase for a reply, such as "CEO at Acme"."""
        if self.role and self.organization:
            return f"{self.role} at {self.organization}"
        return self.role or (f"at {self.organization}" if self.organization else "")


def organization(text: str) -> str | None:
    """A run of capitalized words, such as "Acme", "Initech Labs", or "University of Michigan", else None."""
    words = text.split()
    if words and words[0].casefold() == "the":
        words = words[1:]
    if not words or len(words) > 6 or words[-1] in _ORG_JOINERS:
        return None
    if all(_ORG_WORD.fullmatch(w) or (i and w.casefold() in _ORG_JOINERS) for i, w in enumerate(words)):
        return " ".join(words)
    return None


def _head(word: str) -> str:
    return word.casefold().rstrip(".")


def role(text: str, with_org: bool = False) -> str | None:
    """A role such as "CEO", "senior engineer", or "head of sales", else None. `with_org` says an organization
    follows, which lets a head like "partner" count."""
    words = text.split()
    if not words:
        return None
    of = next((i for i, w in enumerate(words) if w.casefold() == "of"), None)
    if of is not None:  # "head of sales": everything after "of" must be a department
        dept = words[of + 1 :]
        if not dept or len(dept) > 3 or not all(w.casefold() in DEPARTMENTS for w in dept):
            return None
        core = words[:of]
    else:
        core = words
    if not core or len(core) > 4:
        return None
    head = _head(core[-1])
    if head not in ROLE_HEADS and not (with_org and head in ORG_ONLY_HEADS):
        return None
    # A possessive ("Acme's CEO") names an organization, which "ORG's ROLE" reads instead.
    if any(w.casefold() in _FUNCTION_WORDS or not re.fullmatch(r"[^\W\d_][\w&-]*\.?", w) for w in core):
        return None
    return " ".join(words)


def _held_at_org(r: str) -> bool:
    """Whether role `r` is one held at an organization (`ORG_HEADS`), such as "CEO" or "VP of Engineering"."""
    words = r.split()
    core = words[: next((i for i, w in enumerate(words) if w.casefold() == "of"), len(words))]
    return bool(core) and _head(core[-1]) in ORG_HEADS


def _role_and_org(text: str, has_sep: bool) -> dict[str, str] | None:
    """Role and organization from "[the] ROLE [of|at|for ORG]" or "ORG's ROLE"."""
    words = text.split()
    det = bool(words) and words[0].casefold() in _DETERMINERS
    body = words[1:] if det else words
    if not body:
        return None
    if has_sep or det:  # a bare role needs a comma or an article ("Mary Cook" is a name)
        for i, w in enumerate(body):
            if w.casefold() not in _ROLE_PREPOSITIONS or i == 0:
                continue
            after = " ".join(body[i + 1 :])
            if w.casefold() == "of" and all(x.casefold() in DEPARTMENTS for x in body[i + 1 :]):
                continue  # "VP of Engineering": the role goes on
            r, o = role(" ".join(body[:i]), with_org=True), organization(after)
            if r and o and (w.casefold() != "for" or _held_at_org(r)):
                return {"role": r, "organization": o}
        if r := role(" ".join(body)):
            return {"role": r}
    if has_sep and not det and (m := re.fullmatch(r"(.+?)'s\s+(.+)", " ".join(body))):
        r, o = role(m.group(2), with_org=True), organization(m.group(1))
        if r and o and _held_at_org(r):
            return {"role": r, "organization": o}
    return None


def attributes(clause: str, self_intro: bool = False) -> dict[str, str] | None:
    """The role and organization in what follows a name, such as ", CEO of Acme" or ", she works at Acme".
    None when the clause is anything else. `self_intro` says the name is the speaker's own, so "I'm" and "I work"
    are about them."""
    c = clause.strip()
    sep = c[:1] in (",", ";", ":") or c.startswith("- ")
    m = _CLAUSES[self_intro].fullmatch(c.lstrip(",;:-").strip() if sep else c)
    if not m:
        return None
    has_sep = bool(sep or m.group("and") or m.group("be"))
    if m.group("works"):
        o = organization(m.group("org"))
        return {"organization": o} if o else None
    return _role_and_org(m.group("rest"), has_sep)


def _sentence(text: str) -> str:
    return re.sub(r"[\s.!?]+$", "", " ".join(text.replace("\u2019", "'").split()))


def parse(
    text: str,
    valid_name: Callable[[str], str | None],
    leads: str | None = SPOKEN_LEADS,
    anywhere: bool = False,
) -> Tag | None:
    """The tag in `text`: a lead-in from `leads` (none when `leads` is None, for a typed name), one name that
    `valid_name` accepts (it returns the name cleaned, or None), and optionally one attribute clause, with nothing
    left over. With `anywhere`, the lead-in can come after other words, in any sentence of `text`.

    Names are tried shortest first, so "Alice Smith, CEO of Acme" tries Alice first, and "Dana Cole" is one name
    because "Cole" is no attribute clause."""
    sentences = re.split(r"(?<=[.!?])\s+", text) if anywhere else [text]
    for s in sentences:
        s = _sentence(s)
        if leads is None:
            starts = [(s, False)]
        else:
            at = r"(?:^|(?<=\W))" if anywhere else r"^\W*"
            starts = [
                (s[m.end() :], bool(re.fullmatch(rf"(?i:{SELF_LEADS})", m.group("lead"))))
                for m in re.finditer(rf"{at}(?P<lead>(?i:{leads}))\s+{_POSSESSIVE}", s)
            ]
        for body, self_intro in starts:
            if tag := _tag(body, valid_name, self_intro):
                return tag
    return None


def _tag(body: str, valid_name: Callable[[str], str | None], self_intro: bool) -> Tag | None:
    words = body.split(" ")
    for n in range(1, min(MAX_NAME_WORDS, len(words)) + 1):
        raw = " ".join(words[:n])
        if "," in raw.rstrip(","):
            break  # a comma ends the name
        name = valid_name(raw.rstrip(",")) if raw.rstrip(",") else None
        if not name:
            continue
        rest = body[len(raw) :]
        if raw.endswith(","):
            rest = "," + rest
        if not rest.strip(" ,"):
            return Tag(name)
        if attrs := attributes(rest, self_intro):
            return Tag(name, **attrs)
    return None
