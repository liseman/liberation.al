// Object memory page: find where something was last seen, and remember new objects from a frozen camera view.
import { Anchor, Button, Card, Group, Stack, Text, TextInput, Timeline, Title } from '@mantine/core';
import { IconEye, IconTag } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, useEffect, useState } from 'react';
import { Disclosure } from '../dashboard/lib/Disclosure';
import { LoadError, PageEmpty, PageHeader, PageLoading } from '../dashboard/lib/page';
import { viewName } from '../dashboard/review/views';
import { messageOf } from '../shared/http';
import { SafeImage } from '../shared/SafeImage';
import { askWhere, evidenceUrl, getObjects, getStatus, type MemoryObject, type QueryResult, type Status } from './api';
import { Evidence } from './Evidence';
import { timestamp } from './format';
import { type ObjectDraft, takeObjectDraft } from './handoff';
import classes from './objects.module.css';
import { Remember } from './Remember';

export const OVERVIEW_KEY = ['objects', 'overview'] as const;
const DEFAULT_CAMERAS = ['cam0', 'cam1', 'cam2'];

async function overview(): Promise<{ status: Status; objects: MemoryObject[] }> {
  const [status, list] = await Promise.all([getStatus(), getObjects()]);
  return { status, objects: list.objects };
}

/** A compact summary of saved objects and cameras available for a capture. */
export function healthLine(status: Status, objects: number): string {
  const fresh = status.cameras.filter(c => c.available && !c.stale && c.source === 'camera').length;
  return `${objects} ${objects === 1 ? 'object' : 'objects'} · ${fresh}/${status.cameras.length} cameras ready${status.observer_error ? ' · matching unavailable' : ''}`;
}

function Answer({ found }: { found: QueryResult }) {
  const earlier = found.history?.slice(1) ?? [];
  return (
    <>
      {found.evidence && <Evidence sighting={found.evidence} />}
      {earlier.length > 0 && (
        <Disclosure summary="Earlier observations">
          <Timeline role="list" className={classes.observations} bulletSize={18} lineWidth={2} mt="xs">
            {earlier.map(h => {
              const Icon = h.kind === 'manual_label' ? IconTag : IconEye; // named by hand, or matched by sight
              return (
                <Timeline.Item
                  key={`${h.evidence}-${h.captured_at}`}
                  role="listitem"
                  bullet={<Icon size={11} aria-hidden />}
                  lineVariant={h.stale ? 'dashed' : 'solid'}
                >
                  <Anchor href={evidenceUrl(h.evidence)} target="_blank" rel="noopener">
                    {`${viewName(h.camera)} · ${timestamp(h.captured_at)} · ${h.kind}`}
                  </Anchor>
                </Timeline.Item>
              );
            })}
          </Timeline>
        </Disclosure>
      )}
    </>
  );
}

function Item({ o, onFind }: { o: MemoryObject; onFind: (name: string) => void }) {
  return (
    <Group wrap="nowrap" py="sm" style={{ borderTop: '1px solid var(--mantine-color-default-border)' }}>
      <SafeImage className={classes.thumb} src={evidenceUrl(o.template)} alt={`Reference crop for ${o.name}`} />
      <Stack gap={0} style={{ flex: 1, minWidth: 0 }}>
        <Text fw={700}>{o.name}</Text>
        <Text size="sm" c="dimmed">
          {o.mode.replace('_', ' ')}
          {o.last_seen && ` · ${timestamp(o.last_seen.captured_at)}`}
        </Text>
        {o.note && <Text size="sm">{`Your note: ${o.note}`}</Text>}
      </Stack>
      <Button variant="default" onClick={() => onFind(o.name)}>
        Last seen
      </Button>
    </Group>
  );
}

export function ObjectMemoryPage() {
  const client = useQueryClient();
  const [objectDraft, setObjectDraft] = useState<ObjectDraft | null>(null);
  useEffect(() => {
    const incoming = takeObjectDraft();
    if (incoming) setObjectDraft(incoming);
  }, []);
  const data = useQuery({ queryKey: OVERVIEW_KEY, queryFn: overview, refetchInterval: 10_000 });
  const [question, setQuestion] = useState('');
  const find = useMutation({ mutationFn: askWhere });

  const health = data.error
    ? `Object memory unavailable: ${messageOf(data.error)}`
    : data.data
      ? healthLine(data.data.status, data.data.objects.length)
      : 'Connecting…';
  const cameras = data.data?.status.cameras.map(c => c.camera) ?? DEFAULT_CAMERAS;
  const cameraNames = Object.fromEntries((data.data?.status.cameras ?? []).map(c => [c.camera, c.name ?? c.camera]));
  const objects = data.data?.objects;
  const answer = find.isPending
    ? 'Looking up saved observations…'
    : find.isError
      ? messageOf(find.error)
      : (find.data?.answer ?? '');

  const submit = (e: FormEvent) => {
    e.preventDefault();
    find.mutate(question);
  };
  const findObject = (name: string) => {
    setQuestion(name);
    find.mutate(name);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Find waits until something is remembered: with nothing, there's nothing to find (Known design pass).
  const findable = !!objects?.length;
  return (
    // The other Known tabs' layout (Known design pass): on the left the title, Find, and what's remembered, and
    // Remember beside them. On a phone: the title, Remember, then the rest, an order that doesn't wait on the list, so
    // nothing moves when it loads.
    <section className={classes.page} aria-labelledby="objects-title">
      <div className={classes.layout}>
        <div className={classes.head}>
          <PageHeader title="Objects" titleId="objects-title" meta={objects ? objects.length : null} />
          <Text size="sm" c={data.error ? 'var(--lib-bad)' : 'dimmed'} aria-live="polite">
            {health}
          </Text>
        </div>
        <Stack gap="md" className={classes.side}>
          {findable && (
            <Card withBorder padding="md" radius="lg" component="section">
              <Title order={2} mb="sm">
                Find something
              </Title>
              <form onSubmit={submit}>
                <Group gap="sm" align="flex-start">
                  <TextInput
                    style={{ flex: '1 1 180px' }}
                    required
                    maxLength={1000}
                    placeholder="Where did I leave my SSD?"
                    aria-label="Object question"
                    value={question}
                    onChange={e => setQuestion(e.currentTarget.value)}
                  />
                  <Button type="submit">Find last seen</Button>
                </Group>
              </form>
              {!!answer && (
                <Text mt="sm" aria-live="polite" style={{ whiteSpace: 'pre-wrap' }}>
                  {answer}
                </Text>
              )}
              {find.data && !find.isPending && <Answer found={find.data} />}
            </Card>
          )}

          <Card withBorder padding="md" radius="lg" component="section">
            <Title order={2} mb="xs">
              Remembered objects
            </Title>
            <Text c="dimmed" size="sm">
              Last seen is a saved observation, not a live location.
            </Text>
            {data.isPending && <PageLoading label="remembered objects" shape="rows" lines={3} />}
            {data.isError && !data.data && (
              <LoadError
                message="Couldn't load the remembered objects"
                detail={messageOf(data.error)}
                onRetry={() => void data.refetch()}
                busy={data.isFetching}
              />
            )}
            {objects?.length === 0 && (
              <div className={classes.empty}>
                <PageEmpty title="No objects remembered yet" compact>
                  Remember your keys, a bag, or a laptop once from a camera view, then ask here where it was last seen.
                </PageEmpty>
              </div>
            )}
            {objects?.map(o => (
              <Item key={o.id} o={o} onFind={findObject} />
            ))}
            <Disclosure className={classes.about} summary="About object matching">
              <Text c="dimmed" size="sm" mt="xs">
                Clear, distinctive crops work best. A different angle or an obstructed view can prevent a match. Some
                objects can only be saved as a manual sighting.
              </Text>
            </Disclosure>
          </Card>
        </Stack>
        <Remember
          key={objectDraft ? 'incoming-draft' : cameras.join()}
          draft={objectDraft}
          cameras={cameras}
          cameraNames={cameraNames}
          onSaved={() => client.invalidateQueries({ queryKey: OVERVIEW_KEY })}
        />
      </div>
    </section>
  );
}
