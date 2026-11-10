// The shell needs only the unread count, not the Inbox report renderer.
import { Badge, Skeleton } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { researchQuery } from './api';

/** The Inbox tab's label, with how many reports are unread. While that count is still loading, a skeleton the size of
 *  its badge sits where the badge goes (hidden from screen readers: the link's name stays "Inbox"); a failed read
 *  shows no badge, as none unread does. */
export function InboxLabel() {
  const q = useQuery(researchQuery());
  const unread = q.data?.unread ?? 0;
  return (
    <>
      Inbox
      {q.isPending ? (
        <Skeleton
          renderRoot={props => <span {...props} />}
          circle
          height={16}
          ml={4}
          display="inline-block"
          style={{ verticalAlign: 'middle' }}
          aria-hidden="true"
          data-testid="inbox-count-skeleton"
        />
      ) : (
        unread > 0 && (
          <Badge size="xs" circle ml={4} aria-label={`${unread} unread`}>
            {unread}
          </Badge>
        )
      )}
    </>
  );
}
