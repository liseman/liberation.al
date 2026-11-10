# Modern web guidance in this repository

Read `SKILL.md` first, then this file. Where the two disagree about this repository, this file wins.

## Which guidance wins

For the dashboard (`frontend/`), the order is: this repository's conventions in
[../mantine/SKILL.md](../mantine/SKILL.md), then Mantine's own guidance, then these guides. The guides are written
for plain HTML and CSS, so where one says to use a native element or API that a Mantine component already covers (a
`<dialog>`, the Popover API, anchor positioning, a native `<select>`, a custom toast), use the Mantine component.
The rest applies as written: CSS layout and type, performance, media and browser APIs, accessibility, and security.

The browser support policy is Baseline widely available, with the fallbacks the guides give for anything newer. The
dashboard runs in current desktop and mobile browsers; there is no older-browser target to add fallbacks for beyond
that.

## Finding a guide offline first

The guides are all here, under `guides/<area>/<id>.md`, each opening with a title and a one-paragraph summary. Search
them with `grep -ril "<term>" .claude/skills/modern-web-guidance/guides` and read the match. This needs no network
and sends nothing anywhere.

## Using the CLI

`SKILL.md` runs `npx -y modern-web-guidance@latest`. Here, pin the version the guides were copied from and turn its
usage reporting off:

```sh
DISABLE_TELEMETRY=1 npx -y modern-web-guidance@1.0.0 search "<query>"
DISABLE_TELEMETRY=1 npx -y modern-web-guidance@1.0.0 retrieve "<id>"
```

Its `search` ranks guides with an index shipped in the package, so after the first download it works offline. Without
`DISABLE_TELEMETRY=1` it sends Google the agent's name, the guide IDs it retrieved, and the search queries. Keep
queries generic (`"light dismiss dialog"`), never a person, place, or anything from the household's data. Don't run
`legacy-scan` unless asked for a modernization pass.

## Disclosure section

`SKILL.md` asks for a "Modern Web Guidance" section at the end of a reply. In a pull request, put it under the
change it shaped as a line in "What changed", not as its own heading.
