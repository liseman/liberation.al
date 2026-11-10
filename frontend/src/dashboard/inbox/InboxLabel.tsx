// The shell needs only the unread count, not the Inbox report renderer.
import { Badge } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { researchQuery } from './api';

/** The Inbox tab's label, with how many reports are unread. */
export function InboxLabel() {
  const q = useQuery(researchQuery());
  const unread = q.data?.unread ?? 0;
  return (
    <>
      Inbox
      {unread > 0 && (
        <Badge size="xs" circle ml={4} aria-label={`${unread} unread`}>
          {unread}
        </Badge>
      )}
    </>
  );
}
