# How to keep what you taught safe

This guide shows you how to export what you taught the pack to a bundle, check the bundle, load it on another install,
and reset machine-made data without losing what you taught. [PEOPLE.md](PEOPLE.md#moving-taught-state) says what the
bundle carries and what it leaves out.

## Before you begin

- Run the commands on the machine the pack runs on, from the repository.
- Choose a path for the bundle outside the data folder: `pack-reset-catalog --fresh` deletes `taught*.zip` files
  beside the state folder.

## Export what you taught

1. Write the bundle:

   ```sh
   uv run pack-knowledge export --source sqlite --output ~/taught.zip
   ```

   It prints how many records and files the bundle holds, then a count for each collection and folder.
2. Check it:

   ```sh
   uv run pack-knowledge verify --input ~/taught.zip
   ```

   It prints `Verified N records and N attachments.` and fails with the reason when the bundle is damaged or holds
   something unsupported.

`export` fails when there is no `knowledge.sqlite3` to export, and when the store holds a collection the bundle does
not know. It never changes the install. `--state` and `--faces` point it at other folders than `paths.state` and
`paths.faces` (see [config/defaults.toml](../config/defaults.toml)).

The bundle is not a backup of the data folder. Keep measurements (`paths.measurement`) and model weights yourself.

## Load a bundle on another install

1. Copy the bundle to the new install, and check it there:

   ```sh
   uv run pack-knowledge verify --input ~/taught.zip
   ```

2. Import it:

   ```sh
   uv run pack-knowledge import --input ~/taught.zip
   ```

   The records are installed in one step, so a bundle that fails validation changes nothing. Then each taught file
   (the face gallery, voice clones, person references, and the owner's voice sample) is put back where nothing is
   there yet. `--state` and `--faces` point it at other folders.
3. Read the second line of the output: `Taught files: N put back, N already there, N left as they are (a different
   file is there), N kept only in the bundle (settings).`
   - A file that differs is never overwritten and is listed as `differs, not overwritten: <name>`. Compare it with the
     bundle's copy yourself.
   - Settings are never put back. They are in the bundle as `config/pack.toml` and `config/devices.toml`: open the
     zip and copy what you want by hand.

## Reset machine-made data and keep what you taught

Use this after a catalog format change or a model change, or whenever the catalog, the recordings, and everything
derived from them should start again.

1. See the plan. Without `--yes` the command prints what it would do and changes nothing:

   ```sh
   uv run pack-reset-catalog --keep-taught
   ```

2. Run it:

   ```sh
   uv run pack-reset-catalog --keep-taught --yes
   ```

   Run it with the application open. It stops every service, deletes the machine-made data, and starts the services
   that were running. It leaves `config/devices.toml` and each camera's settings as they are. If it stops partway, it
   leaves everything stopped and says how to finish by hand.

To keep each place's maps, plans, walks, and splats too, add `--keep-places` to both commands.

## Start spatial memory again

To reset only what the pack learned about places, follow [Start again](PLACES.md#start-again): it turns Learn
surroundings off, previews with `pack-reset-places`, runs `pack-reset-places --yes`, and restarts the dashboard and
the journal.

## Reset everything, taught state included

Only when the owner asks for it. Export and verify a bundle first, then:

```sh
uv run pack-reset-catalog --fresh
uv run pack-reset-catalog --fresh --yes
```

`--fresh` deletes everything `--keep-taught` does, plus every taught store, and truncates the logs. Import the bundle
afterwards, as above.

## What each reset keeps

| Command | Deletes | Keeps |
|---|---|---|
| `pack-reset-catalog --keep-taught` | The catalog, recordings, the segment store, the vector index, and derived state | The knowledge store, the face gallery, voice references, `paths.voices`, and models |
| `pack-reset-catalog --fresh` | All of that, and every taught store | Settings (`pack.toml`, `config/devices.toml`) and models |
| `pack-reset-places` | Every place's maps, splats, camera registrations, and stored positions | The knowledge store (place names and plans) and everything else |

## Related

- [People and pets](PEOPLE.md#moving-taught-state): what the bundle carries and leaves out.
- [Storage](STORAGE.md): what is stored, and how long it is kept.
- [Upgrading](../MIGRATING.md#resets-and-what-they-keep): the resets in an upgrade.
