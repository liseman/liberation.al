# floorplan (vendored)

The built output (`dist/`) of the floorplan library: a declarative floor-plan renderer and linter. A plan document
goes in; a scaled SVG string, the room schedule, and findings come out. It has no runtime dependencies and touches no
DOM. The Places tab (`src/dashboard/places/`) draws plans with it.

- Upstream: https://github.com/arturalkaim/floorplans
- Commit: `4644b5f6512c04f0fdf2101944e12bab9a894678`
- Version: 0.1.0
- License: MIT, as declared in the upstream `package.json`. The upstream repo has no LICENSE file at this commit.

## What is here

`dist/*.js` and `dist/*.d.ts` as `npm run build` (`tsc -p tsconfig.build.json`) writes them at that commit, unchanged,
except that `bin` and `cli` are left out: they are the Node command line and import `node:` modules.

These files are not edited here. Our own additions to the plan format (provenance, unexplored areas, the anchor
transform, visit history) go in a layer beside this directory, so they can go upstream later. Biome skips this
directory (`biome.json`).

## Updating

In a throwaway directory outside this repo: clone the upstream at the new commit, `npm ci`, `npm run build`, then
replace these files with `dist/` (still without `bin.*` and `cli.*`) and update the commit above.
