# How to ask the pack a question

This guide shows you how to ask a question about your recordings, read the answer, and send a harder question to
Deep research, whose report arrives in the Inbox. [Answers](ANSWERS.md) explains how a question is read and answered.

## Before you begin

- The pack has recorded something to ask about.
- Deep research needs Ask's own model server: the `ask-ollama` service, with `ask.ollama_url` pointing at it
  ([config/defaults.toml](../config/defaults.toml)). Without it the switch is disabled and says so.

## Open Ask

There is one Ask, and it opens the same way from anywhere:

- Press **Ask** in the dashboard's header, or the `/` key.
- Choose it in **Go to** (Ctrl K).
- In Review, open Ask as above. The day Review shows sits over the box as a chip, and the question is about that day
  unless it names another time. Remove the chip to ask about any time.

## Ask a question

1. Type the question under "Your question", or pick one of the suggested questions.
2. Press Enter, or **Ask**.

Ask searches your recordings for the time the question names. Nothing leaves the machine. While it looks, it says
"Looking through your recordings", and the answer appears as it is written.

Say the time in the question, in ordinary words:

| The question says | Ask looks through |
|---|---|
| today, yesterday, last night, this morning, around 3pm, on October 20 | That day or part of it |
| since recording started, ever, so far | Everything recorded |
| since October, since Friday | From then until now |
| in March | That month |
| between October 20 and October 25 | Those days |
| the last 3 weeks, the past 2 months | That long, back from now |
| No time at all, or a time Ask can't read | Everything recorded |

Every answer says what it covered, in a line that begins "Looked through".

"How many times did Dana say thanks" is counted over the transcript for that time, not estimated, and the lines that
say it are listed with the answer.

![Ask open over Live on the demo household, with a question, its answer, and Was this right? under it](images/demo-ask.png)

Under the answer:

- **Warnings** say what the answer could not see, such as a stretch nothing was recorded.
- **Was this right?** takes **Yes**, or **No, correct it**, which asks what a right answer says. Either saves the
  question for checking later answers against.
- **Past questions** lists what was asked before, with the answers.

Opened from Review, the answer's cited moments are listed as times. Click one to play from there.

## When there is no quick answer

When nothing in your recordings answers the question quickly, Ask says so in one line and offers **Try Deep
research**. It offers the same under an answer that rests on only part of the evidence. Press it to send the same
question to Deep research.

A question Ask can't read, such as one that names a camera it can't tell apart from another or a clock time that
conflicts with its date, also comes back as one line that says what it needs, with **Try Deep research**. It is a
notice, not an answer: there is nothing under it to mark right or wrong. Ask it again with what the line asks for.

## Use Deep research

Deep research takes longer and looks deeper: more rounds of searching, at idle priority, one question at a time.

1. Turn on **Deep research** under the question box. The button reads **Research**.
2. Press **Research**.

Ask says "Research started", and about how long a report usually takes on this machine. You can close Ask and carry
on. The switch is off again for the next question: a question is quick unless you ask for more that time.

## Read the report

Open **Inbox**. It lists the questions sent to Deep research, the newest first, with "Waiting" on one that isn't
done. Reports come only from Ask's Deep research switch. Pick a report to read it: the answer, the moments it cites,
and what it looked at.

![The Inbox on the demo household: four questions, one waiting, and a report open with its answer and the three moments it cites](images/demo-inbox.png)

- A cited moment's time opens Review there, and **Open in Review** opens the day the question was about.
- **Ask again** researches the question again, as a new report.
- **Ask a follow-up** opens Ask with the report as a chip over the box. A follow-up gets a quick answer.
- **Mark unread** keeps the report counted beside Inbox in the header.
- **Delete** removes the report for good. Asking the question again starts the research over.

## Related

- [Answers](ANSWERS.md): how a question is read, what counts as evidence, and which questions are counted, not
  written.
- [Activities](ACTIVITIES.md): make a kind of activity count in Ask.
- [Objects](OBJECTS.md): ask where a thing was last seen.
