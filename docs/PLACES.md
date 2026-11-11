# How to see what each camera sees in 3D and people on a place's plan

This guide takes a pack with fixed cameras to pictures collected from them, a point cloud of what each camera sees,
cameras that have a pose from their spot on the plan, and people drawn on the plan on the Places page. [Spatial memory](SPATIAL.md) explains how each part
works.

## Before you begin

- The place has a plan, drawn in the plan editor ([How to draw and change a place's plan](PLAN-EDITOR.md)).
- The place-recognition model is fetched: `mise run models`.
- The `surroundings` and `positions` services are running. Both are on by default
  ([services/services.toml](../services/services.toml)).
- Use the dashboard on the machine the pack runs on: only that machine can change the Learn surroundings switch.

## Turn on Learn surroundings

1. Open the Live tab and turn on the **Learn surroundings** switch.
2. Open the camera button beside the switch, "Cameras to learn from", and check the cameras to collect pictures from.
   With none checked, the default camera (marked "Used when none is chosen" in the menu) is used.

![Live on the demo household, with the Learn surroundings switch and its camera button above the four cameras](images/demo-live.png)

Each change is saved as it is made, and nothing restarts. Turn the switch off to stop. On another device the switch
and the menu are disabled, and say that only the machine the pack runs on can change them.

## Check what the panel shows

Open Known, then Places. While the switch is on, each place carries a "Learning" badge, and the "Learning
surroundings" panel leads with one state:

| The panel says | Meaning |
|---|---|
| Collecting pictures from N cameras | The sampler is looking at the chosen cameras |
| No camera chosen | None of the chosen cameras is in the device manifest: choose again on the Live tab |
| Not collecting pictures | The switch is off; the pictures already kept stay |

The panel is hidden while the switch is off and nothing is kept. Under the state is "Pictures kept", and a row for
each camera with its newest kept picture (with anyone in view painted over; none is shown without the person detector's
model). A row that says "Keeping pictures" needs nothing. "Nothing new in view", "Person or animal in view", "Too
dark", and "Picture is blurred" say why the camera's recent pictures were passed by: wait, or change what the camera
sees. "Not looked at yet" means no picture of the camera has been judged since the sampler started.

## Look at a camera's point cloud

While Learn surroundings is on, each chosen camera also gets a point cloud of what it sees. It needs MASt3R-SLAM set
up on the machine ([the steps](../MIGRATING.md#each-cameras-own-point-cloud)); without it the view says so.

1. On the Places page, pick the place and switch its view from **Plan** to **3D view**.
2. Pick a camera under **Camera**. The view opens from that camera's own viewpoint and draws its points.
3. Press **See it from outside** to step back from the camera and look at the cloud from elsewhere.

![The 3D view of the demo household's front-room camera: the room as a cloud of points, with the camera picker above and the colour, point size, and floor controls below](images/demo-3d-view.png)

**One camera** shows that camera's cloud by itself. **All cameras** puts every camera's cloud where its camera stands
on your plan ([See every camera's cloud in one scene](#see-every-cameras-cloud-in-one-scene)).

The line beside the picker says how many points the cloud has and when it was made. "Approximate size" means the
distances are the model's own estimate, which can be a tenth off or more.

When a camera has no cloud, the view says why:

| It says | Do this |
|---|---|
| Learn surroundings is off, or the camera isn't chosen to learn from | Turn the switch on and choose the camera on the Live tab |
| Waiting for about 20 seconds with nobody in view | Wait |
| Someone was in view each time it looked | Leave the view empty for about 20 seconds |
| The view is too dark, or the camera is in night mode | Light the view |
| The model that makes point clouds isn't set up | Set up MASt3R-SLAM and `spatial.pointmap_runtime` |
| There wasn't enough free GPU memory, or the last try failed | Wait: it tries again in a couple of minutes |
| The model wasn't sure of enough of the picture, or the picture had no depth to speak of | Glare, glass, or a very plain view: change what the camera sees |
| This camera's lens calibration can't be used | Calibrate the lens again ([Cameras](CAMERAS.md#calibrate-a-wide-lens)) |

A cloud is made again when the camera's view changes.

## See every camera's cloud in one scene

1. Give each camera a spot with a facing on your plan
   ([Give a camera a spot on the plan](#give-a-camera-a-spot-on-the-plan)).
2. In the 3D view, switch from **One camera** to **All cameras**.

Each camera's cloud is drawn where its camera stands on your plan. The clouds are separate layers and are never
blended: each camera shows only what it sees, so expect gaps and seams. Under the view is a row for each camera:

- The checkbox shows or hides its points.
- The colour is its outline in the scene, and its points' tint when the points are coloured **By camera**.
- **Look from** moves the view to where the camera stands.
- The arrow at the end opens the details: how the camera was placed, what its pictures share with another camera's,
  and how many of its points fall outside your plan's rooms.

The word on each row says how the camera is placed:

| It says | Meaning | Do this |
|---|---|---|
| Placed | Placed from its spot on your plan | Nothing |
| Adjusted | Placed from its spot, then adjusted against another camera that sees the same space, by what their pictures share. The details say how far from the spot | Nothing |
| Moved | The camera was moved, and its cloud followed to where its pictures now put it | Press **Move its spot** to move the spot on your plan there. It asks first, and says how far the spot moves and turns. **Undo** on the message that follows puts the spot back, and the plan's earlier version stays in Versions |
| Not placed | The camera was moved, and its picture matches neither its old one nor another camera's | Set its spot again in the plan editor |
| No spot | It has no spot on your plan, or its spot has no facing | Press **Set its spot**, which opens the plan editor |

**Only inside the plan** leaves out the points that fall outside your plan's rooms, and says how many.

Two cameras are adjusted against each other only when their pictures clearly show the same space. When they see some
of it but their pictures don't agree well enough, they are placed from their spots only. When their pictures put
them far from where their spots do, the details say that one of the two spots is off, and nothing is moved. Pictures
taken at night often don't match, so a pair is first adjusted in daylight; the adjustment then stays.

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
or its pose in a revision, its fitted tilt, and what it lacks. A camera read through "nothing: no spot and no pose"
places nobody. What a camera lacks, and what to do:

| It lacks | Do this |
|---|---|
| Height, facing, or field of view | Fill in that field of its spot ([steps](PLAN-EDITOR.md#give-a-camera-a-spot)) |
| A held lens and its spot has no field of view | Give its spot a field of view |
| Tilt: too few people seen walking | Wait: the tilt is fitted by a `pack-positions` run once enough people have walked through the view |
| Check: people come out the wrong height | Check the spot's height and field of view, and the camera's lens |

A line also says how many people and animals the camera left unplaced behind a wall from its spot. A few are
expected (a pet on a sofa). The run flags a camera with more than a tenth of its people there: its height, tilt,
lens, or spot is off, or the plan is drawn at another scale.

The `positions` service places people at 20 past each hour. To place them at once:

```sh
uv run pack-positions
```

## Verify

On the Places page, pick the place and a time window: paths are drawn on the plan, and the people and pets seen are
listed under it. A camera that can't place anyone is listed there too, as "`<camera>` can't place anyone yet", with
what it lacks.

## Start again

`pack-reset-places` deletes every place's maps, splats, camera registrations, fitted tilts, and stored positions,
and each camera's point cloud. It keeps the owner's names and plans, spots included, and a place with an owner plan
stays listed with no map. It leaves the kept pictures, and the clouds are made again.

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
- [Spatial memory](SPATIAL.md): how pictures are collected, why a spot outranks a registration, and how plans and
  positions fit together
- [config/defaults.toml](../config/defaults.toml), `[spatial]`: the settings behind the switch
- [Upgrading](../MIGRATING.md#resets-and-what-they-keep): the other resets
