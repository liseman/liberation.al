# How to have the cameras learn a place and see people on its plan

This guide takes a pack with fixed cameras to a place with a learned map, cameras that have a pose, and people
drawn on the plan on the Places page. [Spatial memory](SPATIAL.md) explains how each part works.

## Before you begin

- The Linux machine needs CUDA, and the model runtimes set up: `mise run mast3r`, `mise run gsplat`, and
  `mise run models` ([Model runtimes](SPATIAL.md#model-runtimes)).
- The `surroundings` and `positions` services are running. Both are on by default
  ([services/services.toml](../services/services.toml)).
- Use the dashboard on the machine the pack runs on: only that machine can change the Learn surroundings switch.

## Turn on Learn surroundings

1. Open the Live tab and press **Learn surroundings**.
2. Open the arrow beside the button and tick the cameras to learn from. With none ticked, the default camera (the
   first in `config/devices.toml`) is used.

Nothing restarts. Press the button again to stop.

## Watch the map being learned

Open Known, then Places. While learning is on, each place carries a "Learning" badge and the "Learning
surroundings" panel leads with one state:

| The panel says | Meaning |
|---|---|
| Nothing kept yet | The cameras are watched; no frame was worth keeping so far |
| N frames waiting | Kept frames wait for the next map step, which starts by itself |
| Placing N new views, Training the splat, Saving the map | A map step is running |
| Up to date | Every kept frame is in a map |
| No camera chosen | None of the chosen cameras is in the device manifest: choose again on the Live tab |
| Learning is off | The switch is off; what was learned stays |

Under it are four numbers (views in the maps, and frames kept, waiting, and parked) and a row for each camera. A row
that says "Keeping frames" needs nothing. "Nothing new in view", "Person or animal in view", "Too dark", and "Picture
is blurred" say why the camera's recent pictures were passed by: wait, or change what the camera sees. A row marked
as a warning has a parked frame, which is the next section.

When a place has a map, switch its view from Plan to 3D to see it.

## Act on a parked frame

A parked frame is kept but can't be placed in a map. Rest the pointer on the camera's row for the full reason.

| The row says | Do this |
|---|---|
| No spot on the plan | Give the camera a spot (below), or also learn from a camera whose view overlaps this one's |
| Needs a facing | Add `yaw` to the camera's spot |
| Needs a height | Add `height` to the camera's spot |
| Lens not known yet | Add `fov` to the camera's spot |
| Working out its tilt | Wait: the tilt is fitted by a `pack-positions` run once enough people have walked through the view |
| Setup doesn't fit | People come out the wrong height: check the spot's `height` and `fov` |
| Floor doesn't match | The floor the camera sees isn't at the plan's floor: check the spot's `height` |
| Plan can't hold a map | The plan's frame is mirrored, and a view can't be placed in it from a spot |

A step tries the frame again when a map gains a view or the camera's spot changes. A frame parked for a day is
dropped, and the camera's next frame takes its place.

## Give a camera a spot on the plan

A spot is one line in the place's plan:

```text
x-camera <camera> at <x>,<y> height <m> yaw <deg> [fov <deg>]
```

- `<camera>` is the camera's name in `config/devices.toml`.
- `x,y` are metres in the plan's own coordinates: x grows to the right of the page and y grows down it.
- `height` is the lens's height above the floor, in metres. Measure it with a tape.
- `yaw` is the way the camera faces, in degrees clockwise from up the page (90 looks right).
- `fov` is its horizontal field of view in degrees. Leave it out for a camera with a held lens.

Spots go in with the plan, through `tools/import_owner_plan.py`. The place needs a map with a floor first, so let
learning found one. The page's plan editor (Edit plan) keeps the spots a plan has, and has no control that adds or
changes one.

1. Write the plan in the floorplan DSL (the docstring of `src/liberational/spatial/plan_import.py` gives the
   statements it reads), and the spots in the same file or in one of their own.
2. Run the import as a dry run. It prints the fit, each camera, and a day's positions by room before and after:

   ```sh
   .venv/bin/python tools/import_owner_plan.py --place "Place 1" --plan home.floorplan --extra cameras.floorplan
   ```

3. When the fit looks right, run it again with `--write`.

To change a spot, edit the file and import it again. The tool's docstring has the other options, and `--history` and
`--restore` bring back an earlier plan.

## Check what each camera places

```sh
uv run pack-positions --dry-run
```

It reads the last day (`--hours` changes that), changes nothing, and prints a line for each fixed camera: how many
footprints it saw, how many land on the floor and in a room of the plan, whether it is read through its spot
or its pose in the map, its fitted tilt, and what it lacks. A camera read through "nothing: no spot and no pose"
places nobody.

The `positions` service places people at 20 past each hour. To place them at once:

```sh
uv run pack-positions
```

## Verify

On the Places page, pick the place and a time window: paths are drawn on the plan, and the people and pets seen are
listed under it. A camera that can't place anyone is listed there too, as "`<camera>` can't place anyone yet", with
what it lacks.

## Start again

`pack-reset-places` deletes every place's maps, splats, camera registrations, fitted tilts, and stored positions.
It keeps the owner's names and plans, spots included, and a place with an owner plan stays listed with no map.

1. Turn Learn surroundings off, and pick a moment that isn't 20 past the hour.
2. See what would go and what would stay, then do it:

   ```sh
   uv run pack-reset-places
   uv run pack-reset-places --yes
   ```

3. Restart the dashboard and the journal, which keep what they read:

   ```sh
   uv run pack-services restart dashboard journal
   ```

## Related

- [Spatial memory](SPATIAL.md): why a spot outranks a map, and how maps, plans, and positions fit together
- [config/defaults.toml](../config/defaults.toml), `[spatial]`: the settings behind the switch
- [Upgrading](../MIGRATING.md#resets-and-what-they-keep): the other resets
