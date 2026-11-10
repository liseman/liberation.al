// The rail's "What's happening": the room's log, newest at the bottom, kept scrolled to the newest.
import { Text, Timeline, Title } from '@mantine/core';
import { IconDoorEnter, IconDoorExit, IconMessage, IconWalk } from '@tabler/icons-react';
import { useEffect, useRef } from 'react';
import { ago, hm } from '../lib/format';
import { BlockSkeleton } from '../lib/loading';
import { type HappeningItem, useHappening } from './happening';

const BULLET: Record<HappeningItem['kind'], typeof IconWalk> = {
  arrival: IconDoorEnter,
  departure: IconDoorExit,
  activity: IconWalk,
  talk: IconMessage,
};

export function Happening() {
  const q = useHappening();
  const items = q.data?.items ?? [];
  const list = useRef<HTMLDivElement>(null);
  const atEnd = useRef(true); // following the newest row: false while someone reads further up
  const newest = items.at(-1)?.id;
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new newest row is what moves the scroll
  useEffect(() => {
    const el = list.current;
    if (el && atEnd.current) el.scrollTop = el.scrollHeight;
  }, [newest]);
  const onScroll = () => {
    const el = list.current;
    if (el) atEnd.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };
  return (
    <section className="rail-card" aria-labelledby="happening-h">
      <div className="rail-head">
        <Title order={2} id="happening-h">
          What's happening
        </Title>
        <span className="rail-note">Newest at the bottom</span>
      </div>
      {items.length ? (
        // role: Timeline renders divs, so the list is named a list for screen readers. tabIndex: it scrolls.
        <Timeline
          className="happening"
          ref={list}
          onScroll={onScroll}
          role="list"
          tabIndex={0}
          bulletSize={20}
          lineWidth={2}
          active={items.length - 1}
        >
          {items.map(i => {
            const Bullet = BULLET[i.kind] ?? IconWalk;
            return (
              <Timeline.Item key={i.id} role="listitem" bullet={<Bullet size={12} aria-hidden />}>
                <time dateTime={new Date(i.ts * 1000).toISOString()} title={ago(i.ts)}>
                  {hm(i.ts)}
                </time>
                <span>{i.text}</span>
              </Timeline.Item>
            );
          })}
        </Timeline>
      ) : q.isPending ? (
        <BlockSkeleton lines={3} heading={false} label="Loading what's happening" />
      ) : (
        <Text size="sm" c="dimmed">
          Nothing yet.
        </Text>
      )}
    </section>
  );
}
