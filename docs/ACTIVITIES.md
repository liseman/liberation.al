# How to teach the pack which activities count

This guide shows you how to keep your own list of activities, answer the clips the cameras flag so that a kind of
activity is counted in Ask, and mark an event by hand in Review. The list and the answers are in the dashboard's
**Known** tab, under **Activities**. [ANSWERS.md](ANSWERS.md#activities) explains the rule behind counting.

## Before you begin

- Answers, names, list changes, and marks are accepted from the machine the pack runs on. To teach from another
  device, set `dashboard.remote_teaching` (see [config/defaults.toml](../config/defaults.toml)) and use the dashboard
  password.
- If the rows under "Learning" say "Not counted in Ask" whatever you answer, `events.answer_counts` is off. Turn it
  on to have Ask count activities at all.

## Choose what the cameras look for

Open Known, then Activities. The list starts with a set of everyday activities, and it is yours to change. On a
phone the list is a row of chips above the clips.

To add an activity:

1. Press the **+** button ("Add an activity").
2. Give it a "Name", such as "Squats". The name must not be one already in the list.
3. In "What it looks like", add a few words on the movement, such as "Bending at the knees and standing back up".
   This is optional, and it helps find what to look for.
4. Press "Add".

The confirmation says what the cameras look for, and nothing restarts. If it says "Nothing matches its name yet",
teach the activity from a clip: see [Name a clip in your own words](#name-a-clip-in-your-own-words).

To delete an activity, press the bin on its row ("Delete this activity") and confirm with "Delete". On a phone, pick
the activity's chip first; the bin is beside its progress line. The cameras stop looking for it, and its clips leave
this page and Ask's counts. Its answers are kept: press "Undo" in the confirmation, or add an activity of the same
name again, and they come back. A row that is not one of your activities (such as "Unclassified activity") has no bin.

## Answer the clips

"Clips to check" lists the movements the cameras saw over the last 7 days that have no answer, newest first. Pick a
row in the list to see only that kind's clips, or "All activities" for every kind.

1. Play the clip. The line under it gives the time, the camera, the length, and who it was when a face was matched.
2. Answer "Is this <kind>?":
   - "Yes" when the clip shows that activity.
   - "No" when it does not. A clip answered No is left out of Ask's counts for good, whatever the kind's standing.
   - "Not sure" when you cannot tell. The clip is set aside for this visit and nothing is recorded.

No counts as much as Yes: answer every clip as it is, not only the right ones. The answered clip leaves the list.

## Name a clip in your own words

1. On the clip's card, press "Call it something else".
2. In "Call it", type a name or pick one of your activities, then press Enter.

The field says "Saved". A clip called by one of your activities' names teaches that activity what to look for, which
is how an activity that matched nothing gets started. A name is not an answer: it does not move a kind toward being
counted, so answer Yes or No as well.

## See when a kind starts to count

Each row under "Learning" shows how far that kind is from being counted:

- "N more answers to count": keep answering its clips.
- "More Yes answers to count": enough clips are answered, but too few of them Yes. The kind counts once the share of
  Yes answers is high enough.

The two thresholds are `activity.promotion_min_ratings` and `activity.promotion_min_precision` in
[config/defaults.toml](../config/defaults.toml). A kind that passes moves under "Counted in Ask", with the number of
its clips you confirmed. Then:

- Ask answers "how many times" questions about it. Until then, Ask says it cannot count that kind and how many more
  answers it needs.
- "All activities" shows 1 in 10 of its new clips as a spot check. Pick the kind to see them all.
- Keep answering the spot checks. If the share of Yes answers falls below the threshold, the kind goes back to
  "Learning" and stops being counted.

"Walked past" is counted from the start, and stops only if your answers say it is wrong too often. A camera angle
with enough answers of its own is judged on those answers, not on the kind's as a whole.

## Mark an event by hand

A mark records what you saw happen and how many times, as a label to score the pack's counting against. It teaches
nothing: it changes no clip, no answer, and no count in Ask. Use it for an event the cameras missed or miscounted.

1. Open the Review tab and find the moment.
2. Press "Mark" in the timeline's controls, then drag across the camera's lane from the event's start to its end. Or
   press **I** at the start and **O** at the end while the playhead is there. Escape clears the marks.
3. In the "Mark an event" form, check "From", "To", and "Camera".
4. Fill in "What happened" in your own words, and use the same words each time for the same kind of event.
5. Set "How many times" it happened in the span, counted by watching.
6. Optionally fill in "Who", "Measured" (a name, a number, and a unit), and "Notes".
7. Press "Save". The form confirms with "Saved:" and what you entered.

Marking the same span again replaces its label. [MEASUREMENT.md](MEASUREMENT.md) shows where the labels are kept and
how to score against them.

## Verify

- The kind's row is under "Counted in Ask" in Known, Activities.
- A question in Ask such as "How many times did I sit down today?" gets a count.

## Related

- [ANSWERS.md](ANSWERS.md#activities): how activities are read, and the rule that decides what Ask counts.
- [VISION.md](VISION.md#see-actions): how the cameras find people, actions, and walk-bys.
- [MEASUREMENT.md](MEASUREMENT.md#what-ratings-do-and-do-not-measure): what your answers do and do not measure, and
  how hand-made labels are scored.
- [NAMING.md](NAMING.md): naming the people and pets the clips show.
