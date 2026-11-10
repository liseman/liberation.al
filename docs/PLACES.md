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

1. Open the Live tab and turn on the **Learn surroundings** switch.
2. Open the camera button beside the switch, "Cameras to learn from", and check the cameras to learn from. With none
   checked, the default camera (marked "Used when none is chosen" in the menu) is used.

Each change is saved as it is made, and nothing restarts. Turn the switch off to stop. On another device the switch
and the menu are disabled, and say that only the machine the pack runs on can change them.

## Watch the map being learned

Open Known, then Places. While learning is on, each place carries a "Learning" badge and the "Learning
surroundings" panel leads with one state:

| The panel says | Meaning |
|---|---|
| Nothing kept yet | The cameras are watched; no picture was worth keeping so far |
| N pictures waiting | Kept pictures wait for the next map step, which starts by itself |
| Placing N new views, Building the 3D map, Saving the map | A map step is running |
| Up to date | Every kept picture is in a map |
| No camera chosen | None of the chosen cameras is in the device manifest: choose again on the Live tab |
| Learning is off | The switch is off; what was learned stays |

Under it are four numbers ("In the map", "Pictures kept", "Waiting", and "Can't be placed") and a row for each camera.
A row that says "Keeping pictures" needs nothing. "Nothing new in view", "Person or animal in view", "Too dark", and
"Picture is blurred" say why the camera's recent pictures were passed by: wait, or change what the camera sees. A row
marked as a warning has a parked frame, which is the next section.

When a place has a map, switch its view from Plan to 3D map to see it.

## Act on a parked frame

A parked frame is kept but can't be placed in a map. Rest the pointer on the camera's row for the full reason.

| The row says | Do this |
|---|---|
| No spot on the plan | [Give the camera a spot](#give-a-camera-a-spot-on-the-plan), or also learn from a camera whose view overlaps this one's |
| Needs a facing | Give the camera's spot a facing ([steps](PLAN-EDITOR.md#give-a-camera-a-spot)) |
| Needs a height | Give the camera's spot a height |
| Lens not known yet | Give the camera's spot a field of view |
| Working out its tilt | Wait: the tilt is fitted by a `pack-positions` run once enough people have walked through the view |
| Setup doesn't fit | People come out the wrong height: check the spot's height and view |
| Floor doesn't match | The floor the camera sees isn't at the plan's floor: check the spot's height |
| Plan can't hold a map | The plan's frame is mirrored, and a view can't be placed in it from a spot |

A step tries the frame again when a map gains a view or the camera's spot changes. A frame parked for a day is
dropped, and the camera's next frame takes its place.

## Give a camera a spot on the plan

A spot says where a camera is on the plan, how high, which way it faces, and how wide it sees. Set it in the plan
editor: [How to draw and change a place's plan](PLAN-EDITOR.md#give-a-camera-a-spot) has the steps.

A camera places nobody until its spot has a facing. People are placed from a saved spot at the next hourly
`pack-positions` run; the next section runs it at once.

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

- [How to draw and change a place's plan](PLAN-EDITOR.md): rooms, walls, camera spots, and versions
- [Spatial memory](SPATIAL.md): why a spot outranks a map, and how maps, plans, and positions fit together
- [config/defaults.toml](../config/defaults.toml), `[spatial]`: the settings behind the switch
- [Upgrading](../MIGRATING.md#resets-and-what-they-keep): the other resets
