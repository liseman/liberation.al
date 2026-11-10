// The Reports tab: charts over everything recorded, each mark a link to Review at that moment. The owner
// picks which reports show and in what order; the server keeps that choice (dashboard/reports.py), not the browser.
// A row of chips under the heading jumps to each report, and the move and remove controls wait behind Arrange.
// It loads as its own chunk (App.tsx), so D3 stays out of the main bundle.
import { ActionIcon, Button, Card, Group, Menu, Stack, Text, Title } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { IconArrowDown, IconArrowUp, IconDots, IconX } from '@tabler/icons-react';
import { type ReactNode, useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { PageEmpty, PageError, PageFrame, PageLoading } from '../lib/page';
import { Activity } from './Activity';
import { type ChosenReport, useLayout, useSaveLayout } from './api';
import { Conversations } from './Conversations';
import { Health } from './Health';
import { DeviceReport, PersonActivityReport, WalkbyReport } from './HourlyReports';
import classes from './reports.module.css';
import { Suggestions } from './Suggestions';
import { reportTitle } from './title';
import { WhosAround } from './WhosAround';

const VIEWS: Record<string, (report: ChosenReport, onPins: (pins: string[]) => void) => ReactNode> = {
  whos_around: (r, onPins) => <WhosAround arg={r.arg} pinned={r.people} onPinned={onPins} />,
  conversations: r => <Conversations arg={r.arg} />,
  activity: r => <Activity arg={r.arg} />,
  health: () => <Health />,
  walkbys: (r, onPins) => <WalkbyReport arg={r.arg} pinned={r.cameras} onPinned={onPins} />,
  person_activity: (r, onPins) => <PersonActivityReport arg={r.arg} pinned={r.people} onPinned={onPins} />,
  devices: r => <DeviceReport arg={r.arg} />,
};

/** Where a report's card is on the page, for its jump chip. */
const anchor = (r: ChosenReport) => `report-${r.id.replace(/[^\w-]/g, '_')}`;

/** A row of chips under the heading, one per report, that jumps to its card. */
function JumpChips({ reports }: { reports: ChosenReport[] }) {
  const { ref, height } = useElementSize<HTMLElement>();
  // The row wraps on a narrow window, so the cards' scroll margin reads its real height.
  useEffect(() => {
    if (height <= 0) return;
    document.documentElement.style.setProperty('--reports-jump-h', `${Math.ceil(height)}px`);
    return () => {
      document.documentElement.style.removeProperty('--reports-jump-h');
    };
  }, [height]);
  if (reports.length < 2) return null;
  return (
    <nav ref={ref} className={classes.jump} aria-label="Jump to a report">
      {reports.map(r => (
        <a
          key={r.id}
          href={`#${anchor(r)}`}
          className={classes.chip}
          onClick={e => {
            e.preventDefault();
            const card = document.getElementById(anchor(r));
            card?.scrollIntoView({ block: 'start' });
            card?.focus({ preventScroll: true });
          }}
        >
          {reportTitle(r)}
        </a>
      ))}
    </nav>
  );
}

/** The smaller kinds, two to a row on a wide screen: reports.module.css's .cards grid. */
const HALF = new Set(['person_activity', 'devices', 'walkbys']);

/** The reports that sit half width: two smaller ones next to each other in the page's order share a row; a smaller
 *  one without a smaller neighbour stays full width. */
export function pairsOf(reports: readonly ChosenReport[]): ReadonlySet<string> {
  const out = new Set<string>();
  for (let i = 0; i < reports.length - 1; i++) {
    const [a, b] = [reports[i], reports[i + 1]];
    if (a && b && HALF.has(a.kind) && HALF.has(b.kind)) {
      out.add(a.id).add(b.id);
      i++;
    }
  }
  return out;
}

function ReportCard({
  report,
  paired = false,
  first,
  last,
  arranging,
  onMove,
  onRemove,
  onPeople,
}: {
  report: ChosenReport;
  /** Half width beside its pair on a wide screen (pairsOf). */
  paired?: boolean;
  first: boolean;
  last: boolean;
  arranging: boolean;
  onMove: (by: -1 | 1) => void;
  onRemove: () => void;
  onPeople: (people: string[]) => void;
}) {
  const view = VIEWS[report.kind];
  return (
    <Card
      withBorder
      padding="lg"
      radius="md"
      className={classes.card}
      component="section"
      aria-label={reportTitle(report)}
      id={anchor(report)}
      tabIndex={-1}
      data-kind={report.kind}
      data-arg={report.arg ? '' : undefined}
      data-paired={paired || undefined}
    >
      <Group justify="space-between" mb="sm" wrap="nowrap">
        <Title order={2}>{reportTitle(report)}</Title>
        {/* Removing a card is its own ⋯ menu; Arrange adds Up and Down, and its own Remove. */}
        {!arranging && (
          <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <ActionIcon variant="subtle" color="gray" aria-label={`More for ${reportTitle(report)}`}>
                <IconDots size={16} aria-hidden="true" />
              </ActionIcon>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item color="red" onClick={onRemove}>
                Remove from this page
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        )}
        {arranging && (
          <Group gap={4} wrap="nowrap">
            <ActionIcon
              variant="subtle"
              color="gray"
              disabled={first}
              onClick={() => onMove(-1)}
              aria-label={`Move ${reportTitle(report)} up`}
            >
              <IconArrowUp size={16} aria-hidden="true" />
            </ActionIcon>
            <ActionIcon
              variant="subtle"
              color="gray"
              disabled={last}
              onClick={() => onMove(1)}
              aria-label={`Move ${reportTitle(report)} down`}
            >
              <IconArrowDown size={16} aria-hidden="true" />
            </ActionIcon>
            <ActionIcon variant="subtle" color="gray" onClick={onRemove} aria-label={`Remove ${reportTitle(report)}`}>
              <IconX size={16} aria-hidden="true" />
            </ActionIcon>
          </Group>
        )}
      </Group>
      {view ? view(report, onPeople) : <Text c="dimmed">This kind of report isn't available in this version.</Text>}
    </Card>
  );
}

export default function ReportsTab() {
  const layout = useLayout();
  const save = useSaveLayout();
  // Reordering and removing are an occasional chore, so their controls show only after Arrange.
  const [arranging, setArranging] = useState(false);
  const lede =
    'Built from everything recorded. Hatched stretches have no observations or coverage. Select a mark to open Review there.';
  if (layout.isPending)
    return (
      <PageFrame title="Reports" lede={lede} className={classes.root}>
        <PageLoading label="Reports" lines={8} />
      </PageFrame>
    );
  if (layout.isError)
    return (
      <PageFrame title="Reports" lede={lede} className={classes.root}>
        <PageError
          title="Couldn't load Reports"
          detail={messageOf(layout.error)}
          onRetry={() => void layout.refetch()}
          busy={layout.isFetching}
        />
      </PageFrame>
    );
  const chosen = layout.data.reports;
  const paired = pairsOf(chosen);
  // Built-in reports (their id is their kind) the owner took off the page.
  const removed = Object.entries(layout.data.kinds)
    .filter(
      ([kind]) =>
        (layout.data.built_in ?? ['whos_around', 'conversations', 'activity', 'health']).includes(kind) &&
        !chosen.some(r => r.id === kind),
    )
    .map(([kind, title]) => ({ id: kind, kind, arg: null, title }));
  const move = (i: number, by: -1 | 1) => {
    const next = [...chosen];
    const [r] = next.splice(i, 1);
    if (r) next.splice(i + by, 0, r);
    save.mutate(next);
  };
  const arrange = chosen.length > 0 && (
    <Button
      variant={arranging ? 'light' : 'default'}
      size="xs"
      aria-pressed={arranging}
      onClick={() => setArranging(a => !a)}
    >
      {arranging ? 'Done arranging' : 'Arrange'}
    </Button>
  );
  return (
    <PageFrame title="Reports" lede={lede} actions={arrange} className={classes.root}>
      <JumpChips reports={chosen} />
      <Stack gap="lg">
        {save.isError && (
          <PageError title="The change to this page wasn't saved" detail={save.error.message}>
            Your reports are as they were. Try the change again.
          </PageError>
        )}
        <div className={classes.cards}>
          {chosen.map((r, i) => (
            <ReportCard
              key={r.id}
              report={r}
              paired={paired.has(r.id)}
              first={i === 0}
              last={i === chosen.length - 1}
              arranging={arranging}
              onMove={by => move(i, by)}
              onRemove={() => save.mutate(chosen.filter(x => x.id !== r.id))}
              // The filter's pins: cameras for Walk-bys by camera, people for the others (dashboard/reports.py FILTERED).
              onPeople={pins =>
                save.mutate(
                  chosen.map(x => (x.id === r.id ? { ...x, [r.kind === 'walkbys' ? 'cameras' : 'people']: pins } : x)),
                )
              }
            />
          ))}
        </div>
        {!chosen.length && <PageEmpty title="No reports on this page">Add one from the suggestions below.</PageEmpty>}
        <Card
          withBorder
          padding="lg"
          radius="md"
          className={classes.card}
          component="section"
          aria-label="Suggested reports"
        >
          <Suggestions removed={removed} onAdd={r => save.mutate([...chosen, r])} />
        </Card>
      </Stack>
    </PageFrame>
  );
}
