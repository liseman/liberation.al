# How to draw and change a place's plan

This guide shows you how to draw a place's plan (its rooms, walls, doors, and windows), give each camera a spot on it,
and keep or undo your changes. Everything here is done on the dashboard's **Known** tab, under Places.
[Spatial memory](SPATIAL.md#the-owners-plan-and-the-map) explains how a plan, a map, and the positions fit together.

## Before you begin

- Saving is accepted from the machine the pack runs on. To save from another device, allow teaching there
  ([Allow teaching from another device](REMOTE.md#allow-teaching-from-another-device)).
- A camera can be given a spot only once it is set up ([Cameras](CAMERAS.md)).

## Open the editor

Open Known, then Places, and pick the place. A place with a plan shows a switch, **Plan** and **3D map**, that only
changes how you look at it; neither changes the plan. Press **Edit plan** to change it.

A place with no plan starts an empty one: draw rooms and walls from nothing. If a stored plan can't be read (it is in
an old format), the editor also starts a new one, and says why above the editor. **Edit plan** is off while the page
shows the machine's own plan beside yours (the **Your plan** and **The machine's** switch): you edit your own.

The editor has a **Plan** panel (its levels and rooms), **Build**, and **Cameras**.

## Draw rooms and walls

1. Open **Build** and pick a wall tool from its rooms section, then draw walls on the plan. The editor finds a room
   wherever walls close a shape.
2. To put a door or a window in a wall, pick **Door** or **Window** under **Add to rooms** and click the wall.

The drawing tools and gestures are those of the editor this page is built on. This project adds the **Build** and
**Cameras** panels, and keeps the tools a home's plan needs: rooms, walls, doors, and windows.

A plan needs at least one room before it can be saved. Until it has one, the editor says "A plan needs at least one
room before it can be saved."

## Give a camera a spot

A spot says where a camera is on the plan, how high, which way it faces, and how wide it sees. People are placed from
a saved spot, and a spot with no facing places nobody. Open **Cameras**. Each camera with a spot is a dot, with its
view drawn as a cone, and a button in the list ("(no facing)" marks one without a facing).

- **To add a camera**, pick it under **Add a camera** (the list holds the cameras with no spot), then click the plan
  where it is, in the 2D view with the select tool. Or press **Put it in the middle** and move it afterwards.
  If you see "No cameras are set up yet", add one in Devices first.
- **To move a camera**, use the select tool and drag its dot.
- **To turn it**, drag the handle at the cone's tip. The facing is in degrees clockwise from up the page (0 is up, 90
  is right).
- **To nudge it**, focus the dot or the handle and press the arrow keys. A dot moves 5 cm a press, and 50 cm with
  Shift. A handle turns 5 degrees a press, and 15 with Shift.
- **To type the numbers**, select the camera (its dot, or its button in the list) and fill in the form:

  | Field | Value |
  |---|---|
  | X, m and Y, m | Where the camera is on the plan |
  | Height, m | The lens's height above the floor. Measure it with a tape |
  | Facing, degrees (0 is up) | The way the camera looks |
  | Field of view, degrees | Its horizontal view. Leave it empty ("not set") when the lens is already known |

  A field is written when you leave it or press Enter. Emptying Height, Facing, or Field of view clears it; X and Y
  go back to what they were.
- **To take a camera off the plan**, select it and press **Remove this spot**. It is listed under **Add a camera**
  again.

A parked frame waiting on a height, a facing, or a view is cleared by filling in that field
([Act on a parked frame](PLACES.md#act-on-a-parked-frame)).

## Save or discard your changes

Nothing is saved until you press **Save plan**. While the editor holds changes it says "Unsaved changes". Each save
becomes a version of the place's plan, which now reads "Your corrected plan". After a save, the page closes the editor
and shows the plan.

- **Discard changes** puts the editor back on the last saved plan.
- **Close editor** leaves the editor. With unsaved changes, it asks "Discard your unsaved changes?" first. Picking
  another place asks the same. The **Plan** and **3D map** switch is hidden while the editor is open.
- If a save fails, the message "Couldn't save the plan:" gives the reason. A save from a device that may not teach
  says "plans can only be corrected from this machine (Remote teaching is off)".

People are placed from a saved spot at the next hourly run. [Check what each camera places](PLACES.md#check-what-each-camera-places)
runs it at once.

## Go back to an earlier version

There is no autosave: every version is a deliberate save.

1. Press **Versions**, beside **Save plan**. The list shows each version's age and a summary of what changed, newest
   first, with the current one marked "Current". A version that withdrew the plan can't be opened.
2. Pick a version. It opens in the editor to look at only, marked "Earlier version", with none of the panels that
   change the plan.
3. Press **Restore this version** to make it the plan again, or **Back to the current plan** to leave it as it is.

Restoring saves the version again on top, so the version it replaced stays in the list and can be restored the same
way. A place in the examples has no versions, and nothing drawn on it is saved.

## Related

- [Learn a place and see people on its plan](PLACES.md): learn the map, and act on a parked frame.
- [Spatial memory](SPATIAL.md): what a plan is for, and why a spot outranks a map.
