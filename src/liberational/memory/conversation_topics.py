"""Full-window lexical themes; unknown voices never supply named participants."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

STOP = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "have",
    "what",
    "when",
    "where",
    "which",
    "about",
    "would",
    "could",
    "should",
    "there",
    "their",
    "they",
    "them",
    "then",
    "than",
    "been",
    "were",
    "will",
    "your",
    "you",
    "our",
    "are",
    "was",
    "for",
    "but",
    "not",
    "just",
    "like",
    "know",
    "think",
    "really",
    "okay",
    "yeah",
    "yes",
    "right",
    "well",
    "good",
    "going",
    "want",
    "need",
    "does",
    "did",
    "don't",
    "it's",
    "that's",
    "let's",
    "can",
    "all",
    "some",
    "more",
    "most",
    "much",
    "very",
    "get",
    "got",
    "also",
    "how",
    "why",
    "who",
    "say",
    "said",
    "one",
    "two",
    "now",
    "here",
    "into",
    "out",
    "only",
    "still",
    "something",
    "anything",
    "everything",
    "thing",
    "things",
    "thanks",
    "thank",
    "please",
    "sure",
    "maybe",
    "actually",
    "look",
    "see",
    "make",
    "time",
    "today",
    "yesterday",
    "tomorrow",
    "doing",
    "done",
    "being",
    "getting",
    "having",
    "gonna",
    "wanna",
    "come",
    "came",
    "went",
    "back",
    "around",
    "because",
    "people",
    "mean",
    "little",
    "even",
    "way",
    "other",
    "every",
    "another",
    "first",
    "last",
    "over",
    "under",
    "through",
    "after",
    "before",
    "until",
    "kind",
    "against",
    "better",
    "feel",
    "felt",
    "many",
    "someone",
    "anyone",
    "ever",
    "never",
    "always",
    "start",
    "started",
    "stop",
    "stopped",
    "find",
    "found",
    "hear",
    "heard",
    "saying",
    "talk",
    "lot",
    "let",
    "tell",
    "told",
    "trying",
    "try",
    "pretty",
    "put",
    "take",
    "use",
    "used",
    "able",
    "nice",
    "looks",
    "seems",
    "any",
    "each",
    "both",
    "these",
    "those",
    "has",
    "had",
    "him",
    "her",
    "his",
    "its",
    "she",
    "such",
    "too",
    "down",
    "off",
}


def _terms(text: str, names: set[str]) -> set[str]:
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text.lower())
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", text) if w not in STOP and w not in names and "'" not in w}


def analyse(rows: Sequence[Mapping[str, Any]], query: Mapping[str, Any]) -> dict[str, Any]:
    participants = list(query.get("participants") or [])
    names = {word for p in participants for word in p.lower().split()}
    ordered = sorted(rows, key=lambda r: r["ts"])
    counts = Counter(p for r in ordered for p in r.get("participants_matched", []))
    missing = [p for p in participants if not counts[p]]
    eligible = []
    left = right = 0
    nearby: Counter[tuple[str, str]] = Counter()
    for row in ordered:
        while right < len(ordered) and ordered[right]["ts"] <= row["ts"] + 120:
            other = ordered[right]
            source = str(other.get("conversation_source", other["source"]))
            nearby.update((source, p) for p in other.get("participants_matched", []))
            right += 1
        while left < len(ordered) and ordered[left]["ts"] < row["ts"] - 120:
            other = ordered[left]
            source = str(other.get("conversation_source", other["source"]))
            nearby.subtract((source, p) for p in other.get("participants_matched", []))
            left += 1
        source = str(row.get("conversation_source", row["source"]))
        if all(nearby[source, p] > 0 for p in participants):
            eligible.append(row)
    joint = bool(eligible) and not missing
    if not joint:
        # Repeated words from one voice cannot answer a joint-conversation question.
        eligible = []
    substantive = [(r, _terms(str(r["quote"]), names)) for r in eligible]
    substantive = [(r, t) for r, t in substantive if len(t) >= 2]
    frequency = Counter(t for _, terms in substantive for t in terms)
    ranked = sorted((t for t, n in frequency.items() if n >= 2), key=lambda t: (-frequency[t], t))
    themes: list[dict[str, Any]] = []
    used: set[int] = set()
    evidence: list[dict[str, Any]] = []
    for term in ranked:
        hits = [(i, r) for i, (r, terms) in enumerate(substantive) if term in terms]
        if sum(i not in used for i, _ in hits) < 2:
            continue
        labels = []
        seen: set[int] = set()
        for i, row in (hits[0], hits[len(hits) // 2], hits[-1]):
            if i in seen:
                continue
            seen.add(i)
            evidence.append(dict(row, quote=str(row["quote"])[:1200]))
            labels.append(f"[T{len(evidence)}]")
        related = Counter(t for i, _ in hits for t in substantive[i][1] if t != term)
        companions = sorted((t for t, n in related.items() if n >= 2), key=lambda t: (-related[t], t))[:2]
        themes.append(dict(term=term, related=companions, utterances=len(hits), citations=labels))
        used.update(i for i, _ in hits)
        if len(themes) == 3:
            break
    subject = " and ".join(participants) or "the recorded speakers"
    if joint:
        lead = (
            ", ".join(t["term"] for t in themes) + " are recurring words in the attributed exchanges."
            if themes
            else "No recurring substantive theme is supported by the available transcript."
        )
        lead += (
            f"\n\nRecurring words in recorded exchanges involving {subject} (not semantic topic or duration rankings):"
        )
    else:
        reason = (
            "No attributed speech for " + ", ".join(missing)
            if missing
            else "No nearby exchange with all requested speakers"
        )
        lead = f"I cannot establish what {subject} discussed most. {reason}. Unknown speakers are not assigned to them."
        attributed = ", ".join(f"{p}: {counts[p]}" for p in participants)
        lead += (
            f" Searched {len(ordered)} transcript entries across the requested window "
            f"(attributed entries: {attributed or 'none'}). These labels are not proof of a shared conversation; "
            "media playback can also be mislabeled. A verified partner voice attribution is needed before ranking topics."
        )
    if themes:
        body = " ".join(
            f"{t['term']}"
            + (" / " + " / ".join(t["related"]) if t["related"] else "")
            + f" ({t['utterances']} utterances) "
            + " ".join(t["citations"])
            + "."
            for t in themes
        )
    else:
        body = "No recurring substantive theme is supported by the available transcript." if joint else ""
    return dict(
        answer=(lead + " " + body).strip(),
        evidence=evidence,
        coverage=dict(
            rows_scanned=len(ordered),
            participant_utterances=dict(counts),
            missing_participants=missing,
            joint_exchange_supported=joint,
            eligible_utterances=len(eligible),
            substantive_utterances=len(substantive),
            topic_ranking="distinct substantive utterances per recurring lexical theme, not speaking duration",
            themes=themes,
            partial=True,
        ),
        warnings=[
            "ASR and speaker labels may be wrong; media playback may be mislabeled as a person. Gaps and unknown speech prevent a complete ranking."
        ],
    )
