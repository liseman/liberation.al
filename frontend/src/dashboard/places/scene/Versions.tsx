// Every saved version of a place's plan, newest first, in a popover of the plan editor: when each was saved, what changed
// from the one before, and which one is the plan now. Picking one shows it in the editor without changing anything
// (PlaceEditor.tsx); restoring it saves it again on top, so nothing here is ever rewritten.
import { Badge, Text, UnstyledButton } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { messageOf } from '../../../shared/http';
import { ago, dateTime } from '../../lib/format';
import { RowsSkeleton } from '../../lib/loading';
import { LoadError } from '../../lib/page';
import { type PlanVersion, planHistoryQuery } from '../api';
import classes from '../places.module.css';

export function Versions({
  place,
  viewing,
  onView,
}: {
  place: string;
  /** The version being looked at, or null for the plan as it is now. */
  viewing: string | null;
  /** Look at a version; null goes back to the plan as it is now. */
  onView: (version: PlanVersion | null) => void;
}) {
  const history = useQuery(planHistoryQuery(place));
  const versions = [...(history.data?.versions ?? [])].reverse();
  return (
    <section className={classes.versions} aria-labelledby="plan-versions-h">
      <h3 id="plan-versions-h" className={classes.panelTitle}>
        Versions
      </h3>
      {history.isPending ? (
        <RowsSkeleton rows={3} label="the plan's versions" />
      ) : history.isError ? (
        <LoadError
          message="Couldn't load the plan's versions"
          detail={messageOf(history.error)}
          onRetry={() => history.refetch()}
          busy={history.isFetching}
        />
      ) : versions.length === 0 ? (
        <Text size="sm" c="dimmed">
          No saved version yet. Each save adds one here.
        </Text>
      ) : (
        <ol className={classes.versionList}>
          {versions.map(version => {
            const shown = viewing ? version.id === viewing : version.current;
            return (
              <li key={version.id}>
                <UnstyledButton
                  className={classes.version}
                  aria-current={shown ? 'true' : undefined}
                  disabled={version.withdrawn}
                  onClick={() => onView(version.current ? null : version)}
                  title={dateTime(version.at)}
                >
                  <span className={classes.versionWhen}>
                    {ago(version.at)}
                    {version.current && (
                      <Badge size="xs" variant="light" color="teal" tt="none" fw={500}>
                        Current
                      </Badge>
                    )}
                  </span>
                  <Text size="xs" c="dimmed" component="span">
                    {version.summary}
                  </Text>
                </UnstyledButton>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
