# Pascal Editor's static files

What the plan editor's page (`plan-editor/index.html`, `src/plan-editor/`) fetches at run time, served by the dashboard so
that nothing comes from another origin.

- Upstream: https://github.com/pascalorg/editor, `apps/editor/public/`
- Commit: `515ccd49c3343397b3931d490f185766c4832b12` (the day `@pascal-app/*` 1.1.0 was published)
- License: MIT (the upstream `LICENSE`)

## What is here

- `icons/`: the 65 images the code of `@pascal-app/editor`, `nodes`, `viewer`, and `core` 1.1.0 names under `/icons/`.
- `cursor.svg`: the 2D editor's pointer.
- `material/`: the two materials a new room gets (`concrete/prepared_drywall` on its walls, `wood/woodplank_48` on its
  floor).

The files are unchanged. The Draco and Basis decoders are not here: the build copies them from `three` in
`node_modules` (`vite.pascal.ts`).

## Updating

When the `@pascal-app/*` packages move to a new version (all four together, as exact versions in `package.json`):
list the `/icons/...` paths the new code names (`grep -rhoE "/icons/[A-Za-z0-9_./-]+\.(webp|svg|png)"` over
`node_modules/@pascal-app/{editor/src,nodes/dist,viewer/dist,core/dist}`), copy those files from the upstream
repository at the release's commit, copy `src/plan-editor/pascal.css` again from its two sources, and update the
commit above.
