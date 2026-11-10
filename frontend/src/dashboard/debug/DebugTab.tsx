// The Debug tab: every service's log as one live tail, Papertrail style (issue #111). Each line is prefixed with
// its level and its service, colored by them; services can be hidden, a minimum level hides the lower ones (#240), a
// search filters or highlights, the tail can be paused, and a line's time opens the lines around it.
import {
  Alert,
  Badge,
  Button,
  Container,
  Group,
  Highlight,
  Modal,
  ScrollArea,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  UnstyledButton,
} from '@mantine/core';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FilterChip } from '../lib/FilterChip';
import { PageHeader, PageLoading } from '../lib/page';
import { setPref, usePrefs } from '../lib/prefs';
import classes from './debug.module.css';
import {
  around,
  CAP,
  clear,
  clock,
  colorOf,
  groupServices,
  isLevel,
  LEVELS,
  type Level,
  type LogLine,
  levelCounts,
  pause,
  resume,
  type View,
  visible,
} from './logs';
import { OllamaAlert } from './OllamaAlert';
import { useLogAction, useLogStream } from './stream';

const CONN_TEXT = { connecting: 'Connecting…', open: 'Live', retrying: 'Reconnecting…' } as const;
const LEVEL_TEXT: Record<Level, string> = { debug: 'Debug', info: 'Info', warn: 'Warn', error: 'Error' };
const LEVEL_PREF = 'debugLevel'; // the minimum level, kept per browser with the other display prefs

function Line({
  line,
  services,
  query,
  onTime,
  focused,
}: {
  line: LogLine;
  services: string[];
  query: string;
  onTime?: (id: number) => void;
  focused?: boolean;
}) {
  const time = clock(line.t);
  return (
    <div className={classes.line} data-line data-level={line.lv} data-focused={focused || undefined}>
      {onTime ? (
        <UnstyledButton className={classes.time} onClick={() => onTime(line.id)} aria-label={`Lines around ${time}`}>
          {time}
        </UnstyledButton>
      ) : (
        <span className={classes.time}>{time}</span>
      )}
      <span
        className={classes.level}
        data-level={line.lv}
        data-inherited={line.inherited || undefined}
        title={line.inherited ? 'Level from the line before it' : undefined}
      >
        {line.lv.toUpperCase()}
      </span>
      <Text span c={colorOf(services, line.s)} fw={600} className={classes.service}>
        {line.s}:
      </Text>{' '}
      <Highlight span highlight={query.trim()} className={classes.text}>
        {line.l}
      </Highlight>
    </div>
  );
}

export function DebugTab() {
  const { logs, conn } = useLogStream();
  const act = useLogAction();
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<View['mode']>('filter');
  const saved = usePrefs(p => p[LEVEL_PREF]);
  const min: Level = isLevel(saved) ? saved : 'info';
  const [focus, setFocus] = useState<number | null>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);

  const shown = useMemo(
    () => visible(logs.lines, { hidden, query, mode, min }),
    [logs.lines, hidden, query, mode, min],
  );
  const counts = useMemo(
    () => levelCounts(logs.lines, { hidden, query, mode, min }),
    [logs.lines, hidden, query, mode, min],
  );
  const present = logs.services.filter(s => !logs.missing.includes(s));

  useEffect(() => {
    const el = viewport.current;
    if (el && atBottom.current && shown.length) el.scrollTop = el.scrollHeight; // follow the tail unless scrolled up
  }, [shown]);

  return (
    <Container size="xl" py="md" className={classes.page}>
      <Stack gap="sm" className={classes.content}>
        <PageHeader
          title="Debug"
          meta={
            <Badge variant="light" color={conn === 'open' ? 'teal' : 'gray'}>
              {CONN_TEXT[conn]}
            </Badge>
          }
          actions={
            <Group gap="xs" wrap="wrap">
              <TextInput
                aria-label="Search the logs"
                placeholder="Search"
                value={query}
                onChange={e => setQuery(e.currentTarget.value)}
                size="xs"
              />
              <SegmentedControl
                aria-label="Search mode"
                size="xs"
                value={mode}
                onChange={v => setMode(v as View['mode'])}
                data={[
                  { value: 'filter', label: 'Filter' },
                  { value: 'highlight', label: 'Highlight' },
                ]}
              />
              <Button size="xs" variant="default" onClick={() => act(logs.paused ? resume : pause)}>
                {logs.paused ? `Resume (${logs.held.length} held)` : 'Pause'}
              </Button>
              <Button size="xs" variant="subtle" onClick={() => act(clear)}>
                Clear
              </Button>
            </Group>
          }
        />

        <OllamaAlert />

        {/* Until the stream says which services it has, one loading line holds the place of the chips, the levels,
            and the tail together: drawing the tail first and the chips over it pushed it down by their height. */}
        {logs.connects === 0 ? (
          <PageLoading label="the logs" lines={6} />
        ) : (
          <>
            <Group justify="space-between" align="flex-start" wrap="wrap" gap="xs">
              <section className={classes.groups} aria-label="Services shown">
                {groupServices(present).map(group => {
                  const allShown = group.services.every(s => !hidden.has(s));
                  return (
                    <div key={group.title} className={classes.groupRow}>
                      <UnstyledButton
                        className={classes.groupTitle}
                        onClick={() => {
                          const next = new Set(hidden);
                          for (const s of group.services) {
                            if (allShown) next.add(s);
                            else next.delete(s);
                          }
                          setHidden(next);
                        }}
                        aria-label={`${allShown ? 'Hide' : 'Show'} all of ${group.title}`}
                        title={`${allShown ? 'Hide' : 'Show'} all of ${group.title}`}
                      >
                        {group.title}
                      </UnstyledButton>
                      <Group gap={6} aria-label={group.title}>
                        {group.services.map(s => (
                          <FilterChip
                            key={s}
                            color={colorOf(logs.services, s)}
                            checked={!hidden.has(s)}
                            onChange={on => {
                              const next = new Set(hidden);
                              if (on) next.delete(s);
                              else next.add(s);
                              setHidden(next);
                            }}
                          >
                            {s}
                          </FilterChip>
                        ))}
                      </Group>
                    </div>
                  );
                })}
              </section>
              <SegmentedControl<Level>
                aria-label="Minimum level"
                size="xs"
                value={min}
                onChange={v => setPref(LEVEL_PREF, v)}
                data={LEVELS.map(lv => ({
                  value: lv,
                  label: (
                    <span>
                      {LEVEL_TEXT[lv]} <span className={classes.count}>{counts[lv].toLocaleString()}</span>
                    </span>
                  ),
                }))}
              />
            </Group>

            {logs.missing.length > 0 && (
              <Text size="xs" c="dimmed">
                No log yet for {logs.missing.join(', ')} (not installed on this machine, or not started since).
              </Text>
            )}
            {logs.dropped > 0 && (
              <Text size="xs" c="dimmed">
                Showing the latest {CAP.toLocaleString()} lines; older ones were let go.
              </Text>
            )}

            <ScrollArea
              viewportRef={viewport}
              className={`${classes.tail} ${classes.logArea}`}
              onScrollPositionChange={({ y }) => {
                const el = viewport.current;
                atBottom.current = !el || el.scrollHeight - el.clientHeight - y < 24;
              }}
            >
              <div role="log" aria-label="Service logs" aria-live="off">
                {shown.length === 0 && (
                  <Text c="dimmed" p="sm">
                    {logs.lines.length ? 'No lines match.' : 'Waiting for log lines…'}
                  </Text>
                )}
                {shown.map(line => (
                  <Line key={line.id} line={line} services={logs.services} query={query} onTime={setFocus} />
                ))}
              </div>
            </ScrollArea>
          </>
        )}
      </Stack>

      <Modal opened={focus !== null} onClose={() => setFocus(null)} title="Lines around it" size="xl">
        {focus !== null && around(logs.lines, focus).length === 0 && (
          <Alert color="gray">That line has left the buffer.</Alert>
        )}
        <div className={classes.tail} role="log" aria-label="Lines around it">
          {focus !== null &&
            around(logs.lines, focus).map(line => (
              <Line key={line.id} line={line} services={logs.services} query={query} focused={line.id === focus} />
            ))}
        </div>
      </Modal>
    </Container>
  );
}
