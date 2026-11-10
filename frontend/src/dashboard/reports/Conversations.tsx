// Conversations: who talks and when (lines by person and hour of day), how much each day, the words that come up
// week by week, and who talks with whom (one exchange is two people's lines within a minute of each other).
import { Group, Stack, Text, Title } from '@mantine/core';
import type { Conversations as Data } from './api';
import { useReport } from './api';
import {
  Bars,
  dayName,
  dayTicks,
  GapKey,
  Heatmap,
  hourName,
  Pending,
  RampLegend,
  reviewAt,
  shortDayName,
} from './charts';
import classes from './reports.module.css';
import { ReportState } from './state';
import { ConversationChord } from './visualizations';

const HOURS = Array.from({ length: 24 }, (_, h) => hourName(h));
const HOUR_TICKS = new Set([0, 3, 6, 9, 12, 15, 18, 21]);

/** Words that say little about what was talked about, left out of "Words that come up". The
 *  server already drops stop words (memory/conversation_topics.py); these are the conversational filler it keeps. */
const FILLER = new Set(
  'stuff thing things probably anyway same bunch really actually basically gonna kinda sorta maybe okay yeah yes like just right well know think mean sure also still something anything everything lot lots whatever'.split(
    ' ',
  ),
);

export function finding(data: Data): string {
  const top = data.people[0];
  if (!top) return 'No one was heard talking in this time.';
  const byHour = new Array<number>(24).fill(0);
  for (const p of data.people)
    p.by_hour.forEach((c, h) => {
      byHour[h] = (byHour[h] ?? 0) + c.n;
    });
  const peak = byHour.indexOf(Math.max(...byHour));
  const pair = data.pairs[0];
  const talk = `Most talk happens around ${hourName(peak)}, and ${top.name} says the most (${top.lines} lines).`;
  return pair ? `${talk} ${pair.a_name} and ${pair.b_name} talk with each other most.` : talk;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

/** One word's rise and fall, week by week (a suggested report). */
function WordView({ data, word }: { data: Data; word: string }) {
  const w = data.words.find(x => x.word === word);
  if (!w) return <Text c="dimmed">"{word}" hasn't come up in the days read so far.</Text>;
  const best = w.weeks.reduce((a, b) => ((b.n ?? 0) > (a.n ?? 0) ? b : a), w.weeks[0] ?? { week: '', n: 0, day: null });
  const weeks = w.weeks.map(x => x.week);
  return (
    <Stack gap="sm">
      <Text className={classes.findings}>
        "{w.word}" was said {plural(w.total, 'time', 'times')}, most in the week of {shortDayName(best.week)}.
      </Text>
      <Group justify="flex-end">
        <GapKey />
      </Group>
      <Bars
        label={`How often "${w.word}" was said each week`}
        points={w.weeks.map(x => ({
          key: x.week,
          value: x.n,
          label:
            x.n === null
              ? `Week of ${shortDayName(x.week)}: nothing recorded`
              : `Week of ${shortDayName(x.week)}: said ${plural(x.n, 'time', 'times')}`,
          href: x.n && x.day ? reviewAt(x.day, null) : null,
        }))}
        ticks={dayTicks(weeks)}
        tickLabel={i => shortDayName(weeks[i] ?? '')}
        xTitle="Week"
        yTitle="Times said"
      />
      <Pending pending={data.pending} />
    </Stack>
  );
}

export function Conversations({ arg }: { arg: string | null }) {
  const query = useReport<Data>('conversations', arg);
  return (
    <ReportState query={query}>
      {data => {
        if (arg?.startsWith('word:')) return <WordView data={data} word={arg.slice('word:'.length)} />;
        const days = data.daily.map(d => d.day);
        const words = data.words.filter(w => !FILLER.has(w.word.toLowerCase()));
        const weeks = words[0]?.weeks.map(w => w.week) ?? [];
        return (
          <Stack gap="lg">
            <Text className={classes.findings}>{finding(data)}</Text>
            <Stack gap="xs">
              <Group justify="space-between" wrap="wrap">
                <Title order={3}>Talk by person and hour of day</Title>
                <RampLegend low="Fewer lines" high="More" />
              </Group>
              <Heatmap
                label="Lines spoken by each person in each hour of the day"
                rows={data.people.map(p => p.name)}
                cols={HOURS}
                colTicks={HOUR_TICKS}
                cell={(r, h) => {
                  const p = data.people[r];
                  const c = p?.by_hour[h];
                  const label = `${p?.name}, ${hourName(h)}: ${plural(c?.n ?? 0, 'line', 'lines')}`;
                  return { value: c?.n ?? 0, label, href: c?.day && c.n ? reviewAt(c.day, c.t) : null };
                }}
                xTitle="Hour of day (links open the latest day)"
                yTitle=""
                cellHeight={22}
                maxCellWidth={36}
                minCellWidth={11}
                rowLabelWidth={80}
              />
            </Stack>
            <Stack gap="xs">
              <Group justify="space-between" wrap="wrap">
                <Title order={3}>Lines heard each day</Title>
                <GapKey />
              </Group>
              <Bars
                label="Lines heard each day"
                points={data.daily.map(d => ({
                  key: d.day,
                  value: d.lines,
                  label:
                    d.lines === null
                      ? `${dayName(d.day)}: nothing recorded`
                      : `${dayName(d.day)}: ${plural(d.lines, 'line', 'lines')}`,
                  href: d.lines ? reviewAt(d.day, d.t) : null,
                }))}
                ticks={dayTicks(days)}
                tickLabel={i => shortDayName(days[i] ?? '')}
                xTitle="Day"
                yTitle="Lines"
              />
            </Stack>
            {words.length > 0 && (
              <Stack gap="xs">
                {/* One week can't show change, so until there are two the words are a ranked list (#2621 board,
                    idea 4); filler is left out either way. */}
                {weeks.length < 2 ? (
                  <>
                    <Title order={3}>Words that come up</Title>
                    <ol className={classes.wordList} aria-label="Words said most">
                      {words.map(w => (
                        <li key={w.word}>
                          <span>{w.word}</span>
                          <span className={classes.wordCount}>{plural(w.total, 'time', 'times')}</span>
                        </li>
                      ))}
                    </ol>
                  </>
                ) : (
                  <>
                    <Title order={3}>Words that come up, week by week</Title>
                    <Heatmap
                      label="How often each top word was said, by week"
                      rows={words.map(w => w.word)}
                      cols={weeks.map(shortDayName)}
                      colTicks={dayTicks(weeks)}
                      cell={(r, c) => {
                        const w = words[r];
                        const wk = w?.weeks[c];
                        const when = `week of ${shortDayName(wk?.week ?? '')}`;
                        if (!wk || wk.n === null)
                          return { value: null, label: `${w?.word}, ${when}: nothing recorded` };
                        return {
                          value: wk.n,
                          label: `${w?.word}, ${when}: said ${plural(wk.n, 'time', 'times')}`,
                          href: wk.n && wk.day ? reviewAt(wk.day, null) : null,
                        };
                      }}
                      xTitle="Week"
                      yTitle=""
                      cellHeight={18}
                      // A few weeks spread across the card instead of a sliver at its left edge.
                      maxCellWidth={Math.min(200, Math.max(40, Math.floor(640 / Math.max(1, weeks.length))))}
                      minCellWidth={6}
                      rowLabelWidth={110}
                    />
                  </>
                )}
              </Stack>
            )}
            {data.pairs.length > 0 && (
              <Stack gap={6}>
                <Title order={3}>Who talks with whom</Title>
                <ConversationChord pairs={data.pairs} />
              </Stack>
            )}
            <Pending pending={data.pending} />
          </Stack>
        );
      }}
    </ReportState>
  );
}
