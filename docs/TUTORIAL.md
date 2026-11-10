# A first look at the pack

In this tutorial we open the dashboard on a made-up household (Alice, Dana, Sam, and Rex the dog) and read one day
of their life in the Journal: the day's page, one hour, and that hour's events. Then we meet the people and the pet
the pack knows, correct one mistake in what it learned about Rex, and find the floorplan of their house. It takes
about ten minutes.

Everyone here is invented. The demo keeps its data in a temporary directory of its own and never reads or writes
`data/`, so nothing we do can touch a real recording.

## Before we start

- Finish "Install" in the [README](../README.md#install), down to `mise run continuum-release`. The demo needs no
  camera, no microphone, no `pack.toml`, and no running Ollama.
- Open a terminal in the clone.

## Step 1: start the demo

```sh
mise run demo
```

The terminal fills with log lines, and among them the command prints two lines like these (the date is
yesterday's, and the directory has a name of its own each time):

```text
pack-demo: seeded the demo household, with a journal and a transcript for 2026-11-08
pack-demo: the demo dashboard is on http://localhost:8780/ (its files are in /tmp/pack-demo-abc123)
```

Leave the command running, and open <http://localhost:8780/> in a browser. The terminal keeps logging while the demo
runs: we can ignore it. If the command says port 8780 is taken, it names the port it took, and the address line
has it: open that address.

The dashboard opens on Live. Notice the status at the top right: "Not listening · cameras off". That is right: this
household has no microphone and no camera, only what was written down about its day.

Live also shows a "Getting started" list, with steps such as picking a camera and recording a voice. Those are for
a pack of our own. Leave them alone here.

If the command stops with `no broker at ...`, the install isn't finished: run `mise run continuum-release`.

One more thing to leave alone: Ask, at the top. The demo runs no models, so a question ends in "Failed".

## Step 2: find yesterday in the Journal

Click **Journal** in the row of tabs.

The Journal is a row of columns, with no headings. The first lists months, with this month picked, and the second
lists that month's weeks.

1. In the weeks column, click the top row, this week. It reads like "Nov 9 to 15".
2. A column of days opens. Click yesterday: the one row that reads `12 h`.

If today is a Monday, yesterday is in the next week down. If it is also the 1st, click last month first.

Two things appear. A column of hours lists twelve hours, from 07:00 to 22:00, each with the start of what the pack
wrote about it. On the right is the day's page:

- **Places:** Demo house (Kitchen, Study, Living room)
- **In short:** a rainy day at home
- **What happened:** five lines, each with its hour
- **Talked about** and **Ate**

Notice the dot beside `12 h`: it marks a day whose page is written. The other days have no dot and no hours,
because nothing happened in this household before yesterday.

## Step 3: open an hour and its events

1. In the hours column, click **18:00** (scroll the column if it is out of sight).

   A column of events opens with the four things that happened in that hour, at 18:02, 18:15, 18:31, and 18:38. The
   pane on the right shows the hour, "18:00 to 19:00", and its entry: Alice cooked pasta while Sam set the table.

   Under the entry is **Evidence**, which reads "Nothing cited." In a pack of our own, this is where the recordings
   an entry came from are listed. The demo has no recordings.

   The pane also has a link, "Where, on the plan". Leave it for now: we find the plan in Step 6.

2. Click the event at **18:31**. The pane shows that one moment: Dana arrived and said hello to Rex in the hall.

3. Press the Left arrow key, then the Up arrow key.

   The arrow keys move through the columns as the mouse does. Left stepped back to the hours, and Up picked 17:00, so
   the pane shows that hour. The events column reads "No events written for this hour."

We have gone from a month down to one moment of a day. Every level of the Journal works this way.

## Step 4: meet the household

Click **Known** in the row of tabs.

Known has a tab for each kind of thing the pack knows by name: People, Pets, Objects, Activities, Devices, and
Places. It opens on People, which lists Alice, Dana, and Sam. Beside the list, "Who's this?" says nobody is waiting
for a name.

Click **Dana**. Her card opens. A card is where everything the pack knows about one person is kept and corrected.
Notice "Voice known" at the top: the pack can tell Dana's voice from the others'. Further down, the card says it
doesn't know her voice yet for a voice clone, which is a different thing: a voice to speak with.

## Step 5: correct what the pack learned about Rex

1. Click the **Pets** tab, then **Rex**. His card opens. It names a camera, demo-cam, and videos that can't be shown:
   the demo's records name a made-up camera, and there is no video behind them.
2. Find **Training images** on the card, and click **Show**.

   The card says "7 examples, the least like the rest first: a wrong one is usually near the top." There are seven
   tiles. Six are coloured and numbered R1 to R6. The first is grey, with a question mark: it is the one that
   doesn't belong. (The tiles are drawings. The demo holds no picture of anyone.)

3. Tick the box on the grey tile, and click **Remove 1 selected**.
4. The dashboard asks "Remove 1 training image from Rex?" Click **Remove**.

The card now says "6 examples", and the grey tile is gone. Under Recognition, the card says Rex is recognized by
appearance, checked on the six that are left.

That is how we correct the pack. It recognizes Rex by matching what it sees against these pictures, so we took out
the one that would have misled it.

## Step 6: find the floorplan

Click the **Places** tab.

The list on the left has one place: Demo house, "5 rooms · 76.5 m² · Nothing seen yet". Beside it is the plan of
the house, with its five rooms: Kitchen, Hall, Bedroom, Living room, and Study. Beside the place's name, at the top
right, the page calls it a "Machine-made plan".

Notice "Nothing seen yet". In a pack with cameras, the paths that people and pets took are drawn on this plan. This
household has no cameras, so nothing is drawn on its plan. For the same reason the page says the place has no map
yet, and how to start one: that is for a pack with cameras.

## Step 7: stop the demo, and start it again

1. In the terminal, press Ctrl-C. The dashboard and the demo's broker stop.
2. Run it again:

   ```sh
   mise run demo
   ```

3. Reload the browser page (at the new address, if the demo printed a different port), open **Known**, **Pets**,
   **Rex**, and view the training images.

Rex has seven examples again, the grey tile among them. Each run makes a new directory and a fresh household, so we
can repeat any step here as often as we like.

Press Ctrl-C once more to finish.

## What we did

We opened the pack's dashboard with nothing at stake, read a day in the Journal from its page down to one event,
opened a person's card, removed a wrong training image from a pet, and found a place's floorplan. A pack of our own
has the same pages, filled from its cameras and microphones.

## Next steps

- Make it yours: [Configure](../README.md#configure) and [First run](../README.md#first-run) in the README.
- Why the Journal reads as it does: [Answers](ANSWERS.md).
- How people, voices, faces, and pets are named and corrected: [People](PEOPLE.md).
- How a place gets its plan, and how people are placed on it: [Spatial memory](SPATIAL.md).
- Every document, by what you came to do: the [documentation map](README.md).
