# How to count visits to a spot with zones

This guide shows you how to outline a spot on a camera's picture (a doorway, a desk, a litter box) as a named zone,
change or remove it, and find the visits counted in it. [VISION.md](VISION.md#zones-and-came-into-view) explains how
visits are found, and [ANSWERS.md](ANSWERS.md#deterministic-answers) how they are counted.

## Before you begin

- Zones are saved from the machine the pack runs on. To save them from another device, set
  `dashboard.remote_teaching` (see [config/defaults.toml](../config/defaults.toml)) and use the dashboard password.
- Visits come from the see-zones service, which follows the people and animals see-actions tracks. Both run by
  default. An animal's visit is labelled with its species.

## Open a camera's zones

1. Open **Settings**, **Devices**.
2. On the camera's row, under "Picture and zones", press **Zones**. The editor opens on the camera's live picture,
   with its saved zones outlined and named.

The picture here is never mirrored, even when **Mirror** is on: a zone belongs to the camera's own frame.

## Draw a zone

1. Press **Add a zone**.
2. Click each corner of the spot on the picture, in order around it. Without a mouse, move the crosshair with the
   arrow keys (Shift for bigger steps) and press Enter or Space to put a corner there. "Undo corner" takes back the
   last one, and "Cancel" drops the outline.
3. Press **Done**. It is enabled once the outline has at least three corners, does not cross itself, and is not too
   small. A zone has at most 32 corners.
4. Type the zone's name in "Name of zone N". Every zone on the camera needs a name, and no two may share one.
5. Press **Save zones**. The editor says "Saved." and the camera's new set takes effect within a few seconds.

A visit is counted from where a person's feet or an animal's paws are: the bottom middle of their box. Outline the
floor or surface someone stands on or at, not the object above it. A camera holds at most 16 zones.

## Mark a TV

A TV in the picture can show people the pack would otherwise count. Press **Draw the TV** and outline its screen. The
new zone is named TV and has "A screen" checked. Nobody inside a screen zone is counted, and while its picture moves,
what is heard can be marked as TV. To make an existing zone a screen, check "A screen" on its row.
[VISION.md](VISION.md#people-on-screens) explains what a screen keeps out.

## Change or remove a zone

Each zone has a row under the picture.

- To rename it, edit its name. The zone keeps its identity, so Ask counts its earlier visits under the new name.
- To reshape it, press **Redraw**, click the new corners, and press **Done**.
- To delete it, press **Remove**.

Then press **Save zones**. Nothing changes until you save, and closing the editor drops unsaved edits. Saving ends
every visit open on that camera; visits start again under the new set.

If the save fails with "these zones changed since you opened them", the zones were saved from somewhere else in the
meantime. Close the editor, open it again, and redo your edit.

## See earlier versions

Press **Earlier versions** at the bottom of the editor. Each save is listed with its time and its zones' names, newest
first. The list is a record only: to go back to an earlier set, draw it again.

## Find the counts

A count takes visits of at least 10 seconds, and joins visits to the same zone less than 30 seconds apart into one.

- **Ask**: ask a count question that names the zone, such as "how many times did someone use the desk today". When no
  footage of the zone was checked in that time, the answer says it can't count, never zero. A question that names a
  person counts only the visits whose faces were recognized as them, and reports the rest apart.
- **Reports**: the "Activity and events" card has a "Zone visits" section with one chart per zone: visits each day,
  against a typical day. Under "Suggested reports", a zone with visits is offered as "Visits to <zone>"; press
  **Add** to give it a card of its own. "Pet activity by hour of day" charts each animal's zone visits by hour.
- **Journal**: a zone visit used as evidence for an entry is listed as "In a zone".

If a question names both a zone and an event the detector or you have named, the event's count answers it.

## Verify

1. Have someone stand in the zone for longer than 10 seconds, then step out.
2. Open **Reports**. In "Activity and events", "Zone visits" shows the zone with a visit today. "Still reading" under
   the card means some days are not counted yet.
   - "Nothing stayed in a zone long enough to count." means the camera was checked, but no visit lasted 10 seconds.
   - "No zones are drawn yet, or none have been checked." means see-zones has covered none of the window: check that
     see-zones and see-actions are running ([Running](RUNNING.md)).

## Related

- [VISION.md](VISION.md#zones-and-came-into-view): how see-zones turns tracks into visits and coverage.
- [ANSWERS.md](ANSWERS.md#deterministic-answers): how a zone count is answered.
- [Cameras](CAMERAS.md): add a camera and name its view.
- [COMPONENTS.md](COMPONENTS.md): what see-zones reads and writes.
