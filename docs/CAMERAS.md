# How to add a camera and calibrate its lens

This guide takes a camera from plugged in to recorded, named, and (for a wide or fisheye lens) calibrated. Do it on
the machine the pack runs on: only that machine can change cameras.
[CONTINUUM-CAPTURE.md](CONTINUUM-CAPTURE.md) explains how capture treats devices.

## Add a USB camera

1. Plug the camera in.
2. Open **Settings**, **Devices**. The **Cameras** card lists every camera the machine sees.
3. Turn on **Use** for the camera. That adds a `[[device]]` for it (named `cam0`, `cam1`, and so on) to
   `config/devices.toml` and restarts capture, so every camera pauses for a few seconds.
4. Open the Live tab and check that its picture is there.

To write the entry by hand, list what capture sees and add a `[[device]]` with `kind = "camera"` and a `match` rule
([config/devices.example.toml](../config/devices.example.toml) documents every key), then restart capture:

```sh
uv run continuum-capture --list
uv run pack-services restart continuum-capture
```

## Add a network camera

Follow [Network cameras](NETWORK.md#network-cameras): discover it in Settings, or enter its RTSP URL. Capture takes
H.264 only.

## Set its picture

On the camera's row in the **Cameras** card:

- **Resolution** picks the size and rate (Linux only, and only when the camera offers more than one). Only that
  camera's worker restarts.
- **Upside down** turns the picture for a camera mounted that way. **Mirror** mirrors it in this browser only.
- **Zones** opens the editor for the zones drawn on the camera's view.
- **Automatic exposure** (Linux only) steers the exposure for the faces in view, or the whole frame when there are
  none. **Target brightness**, from 60 to 180, sets what it steers toward.

## Name its view

A camera is a view, not a room: name it for what it sees. The name shown in People and Review comes from
`vision.camera_names` in `pack.toml`, one `"<camera>: <name>"` for each camera (the **Name** row on the camera's row
in Settings writes the same setting):

```toml
[vision]
camera_names = ["facecam: desk view"]
```

`<camera>` is the device's `name` in `config/devices.toml`. That name is the camera's id everywhere (streams, zones,
places, and labels), so to change the id itself use `pack-rename-camera`, with the services stopped. Its dry run
says what would move and which settings to edit by hand:

```sh
uv run pack-rename-camera facecam=desk --dry-run
```

## Pause it for privacy

Open the status menu in the dashboard's header and choose **Pause all cameras**, or, with several, **Pause** and the
camera's name. Capture stops a paused camera within about a second and records nothing from it until it is resumed.
The others keep running.

## Say how it is mounted

A fixed camera's height puts its place's map at metric scale, and an outdoor camera is kept out of indoor places. Add
`mount` to the camera's `[[device]]`:

```toml
mount = { height_m = 1.5, outdoor = false }
```

`height_m` is from the floor to the lens. A camera carried on a moving rig takes `rig`, `at`, `turn`, and `hfov_deg`
instead ([config/devices.example.toml](../config/devices.example.toml) has an example).
[Places](PLACES.md) covers putting the camera on a place's plan.

## Calibrate a wide lens

A wide lens bends straight lines, which throws off maps and positions. `tools/calibrate_lens.py` fits the bend from
edges that are straight in the room, with no checkerboard.

1. Save a frame from the camera, and write a JSON list of rough polylines along edges that are straight in the room
   (a door frame, a wardrobe's side), each as `[[x, y], [x, y], ...]` in the frame's pixels. Long edges near the
   frame's sides tell the most.
2. Fit, and read the scores. Nothing is written:

   ```sh
   uv run tools/calibrate_lens.py --view desk.png:desk-lines.json
   ```

3. Write the calibration to the camera's entry, with a `--mask` for each overlay the camera burns into its picture
   (a clock or a logo), as fractions of the frame:

   ```sh
   uv run tools/calibrate_lens.py --view desk.png:desk-lines.json \
       --write config/devices.toml --camera facecam --mask 0.25,0,0.74,0.09
   ```

4. Restart capture so the manifest is read again. The next map built from the camera uses the calibration.

The tool's docstring explains the models it compares and its other options.

## Calibrate a fisheye lens

A fisheye camera is calibrated from views of a printed checkerboard (9 by 6 inner corners by default) on something
flat and stiff, so that people in its picture are straightened before they are analyzed.

1. Stop the camera's capture so the device is free, and capture about 20 views. Move the board slowly, near and far,
   tilted each way, and out to the picture's edges and corners:

   ```sh
   .venv/bin/python tools/calibrate_fisheye.py capture --device /dev/video0 --out calib/fish0
   ```

2. Solve. It prints the fit's error in pixels: under 1 is good. Over 1, capture again with more views near the rim.

   ```sh
   .venv/bin/python tools/calibrate_fisheye.py solve calib/fish0 --camera fish0
   ```

3. Run it again with `--write` to set `fisheye` on the camera's entry in `config/devices.toml`.
4. Restart see-actions. Its log says the camera is "a fisheye, rectified per person".

   ```sh
   uv run pack-services restart see-actions
   ```

Calibrate at the size the camera streams at, and once for each lens: two copies of one lens model still differ.
