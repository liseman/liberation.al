Read AGENTS.md before changing anything in this repo, and docs/CONVENTIONS.md before writing anything that will be published (a commit message, an issue, a pull request, or a comment): this repository is public.

Before writing or reviewing tests, load the python-tests skill (`.claude/skills/python-tests/`), and the property-based-testing skill when a test should cover a whole input domain. The code is typed Python: every function you add or change in `src/` and `tools/` needs type hints, and `mise run lint` (Ruff, then the ty type checker) must pass before you commit.

Do not edit `docs/` or `README.md`: put what changed in a "Docs:" section of the pull request (AGENTS.md, "Write things down in the right place"). Before touching documentation at all, load the diataxis skill (`.claude/skills/diataxis/`: `SKILL.md`, then `PROJECT.md`).
