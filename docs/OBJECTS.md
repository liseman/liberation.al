# How to teach the pack an object and ask where it was last seen

This guide shows you how to outline an object (keys, a bag, a laptop) on a camera's picture, name it, and ask where it
was last seen. It is done in the dashboard's **Known** tab, under **Objects**.
[VISION.md](VISION.md#captions-and-objects) explains how the pack finds an object again.

## Before you begin

- Matches come from the see-objects service, which runs by default. With no object taught, it does nothing.
- The camera must be sending a picture. The page counts the cameras that have a fresh one ("1/2 cameras ready").
- Keep faces out of the outline. An outline that overlaps a face seen at that moment is refused.

## Freeze a view to outline from

Pick one way to get a still picture of the object.

- **From the page**: open Known, then Objects. Under "Remember an object", pick the camera in the box (it shows the
  camera's name), wait for its live preview, and press **Freeze view**.
- **From Live**: open a camera's large view and click the box the camera drew around the object. The Objects tab opens
  with that picture frozen and the object already outlined. Go on within a minute of the click.
  A box around an animal opens Pets instead, and one around a face opens People.
- **By voice**: say "remember this as my wallet". The pack saves the view and, for another camera, takes "on camera 2"
  at the end of the sentence. On the Objects page, press **Use what I said**. The name is filled in. The saved view
  lasts 10 minutes.

A frozen view is kept for 10 minutes. After that, freeze it again.

## Outline the object

The frozen picture says "frozen, not live". Do one of these.

- Click the object. The pack outlines it and says "Outlined. Name it, then remember it."
- Press **Crop mode** and drag a box around the object. **Find object in box** outlines the object inside it.
  **Cancel crop** drops the box.
- Press **Find objects** to have the pack outline everything in the part of the picture you can see, then click the
  outline you want. **Zoom in** first on a busy view: it is faster.
- For an object that will not outline, draw a box and press **Use this crop** (at least 24 by 24 pixels). That
  remembers the rectangle as it is.

Outlining takes a few seconds. A new click, box, or search replaces the earlier one.

## Name it and remember it

1. Type the name in "Object name" (up to 80 characters), such as "my SSD".
2. Add "Other names" if you call it more than one thing ("drive", "external SSD"). Press Enter after each. A note
   ("Silver drive with the orange sticker") is optional.
3. Press **Remember selected object** (**Remember crop** after **Use this crop**).

A name or other name that is already taken is refused, and so is a name that says person, face, man, woman, or child.
The page says "Remembered. Looking for conservative visual matches in future frames." when the object has enough
detail to be recognized later. Otherwise it says it is remembered as a manual sighting: it is listed and can be asked
about, but the crop lacks the detail for automatic matching. Clear, distinctive outlines work best.

## Ask where it was last seen

Once one object is remembered, "Find something" appears at the top of the Objects tab.

1. Type a question that names the object or one of its other names, such as "Where did I leave my SSD?", and press
   **Find last seen**. Or press **Last seen** on the object's row under "Remembered objects".
2. Read the answer. It shows the camera, the time, and the picture with the match boxed. "Earlier observations" lists
   older sightings, each a link to its picture.

The pack looks back a day first, then a month. It says "Which object do you mean?" when the question names more than
one, and that it has no evidence when it names none. An answer is an observation, not a live location: it says it
cannot confirm the object is still there.

You can also ask by voice: "where did I leave my wallet?" answers with the camera, the time, and where the picture is.
[ANSWERS.md](ANSWERS.md#deterministic-answers) lists how this is answered.

## Rename or remove an object

Under "Remembered objects", find the object's row.

1. To rename it, press the pencil beside its name. The name box opens with the current name; type the new one, then
   press Enter or the tick. Escape or the cross leaves it as it was. A name, or one of its other names, that another
   object already goes by is refused, and the box stays open so you can change it.
2. To remove it, press the bin. A prompt asks first; press **Remove**.

The new name is what Find and Last seen answer for from then on, and a removed object stops being matched and leaves
the list. Neither touches what was recorded: sightings recorded before a rename still show the name they were
recorded with, and a removed object's sightings stay. The message after a removal offers **Undo**, which remembers the
object again as it was, with its picture, note, and other names. Remembering it again from a camera view also puts it
back.

## Verify

1. Under "Remembered objects", the object is listed with its picture and, for a visual one, "visual".
2. After the object has been in view of a camera for a while, ask "Where did I leave <name>?". The answer names
   a sighting at that time. If its newest sighting is the one you made by hand, the object has not been matched in
   video: check that see-objects is running ([Running](RUNNING.md)), and that the object is in plain view.

## Related

- [VISION.md](VISION.md#captions-and-objects): how outlines and matches work.
- [ANSWERS.md](ANSWERS.md#deterministic-answers): how "last seen" is answered.
- [NAMING.md](NAMING.md): name people and pets, which Objects does not take.
- [DASHBOARD.md](DASHBOARD.md): the Known tabs.
