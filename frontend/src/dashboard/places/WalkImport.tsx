// The Places page's walk import (#176, #192): drop the zip Stray Scanner exports (or choose it) and it's uploaded,
// then imported in the background (dashboard/walk_import.py), as `pack-walkthrough <folder> --match` does by hand.
// Nobody picks a place: the import registers the walk against every place's map and puts it in the one it fits, or
// makes a new place when it fits none, and this says which. One import runs at a time, and the server keeps the
// newest one, so its progress and its result are still here after a reload. When it finishes, the places are read
// again and the new walk shows.
import { Alert, Anchor, Button, FileButton, Group, Progress, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type DragEvent, useEffect, useRef, useState } from 'react';
import { messageOf } from '../../shared/http';
import { dismissWalk, retryWalk, uploadWalk, type WalkJob, walkActive, walkQuery } from './api';
import classes from './places.module.css';
import { WALKTHROUGH_DOC } from './walks';

const VIDEO = ['.mp4', '.mov', '.m4v'];

/** Why a file can't be a walk, before any of it is sent; null when it can. */
export function refusal(file: { name: string; size: number }, maxMb: number, video: boolean): string | null {
  const name = file.name.toLowerCase();
  const isVideo = VIDEO.some(s => name.endsWith(s));
  if (!name.endsWith('.zip') && !isVideo)
    return video
      ? 'Drop the zip Stray Scanner exports, or a video of the walk.'
      : 'Drop the zip Stray Scanner exports.';
  if (isVideo && !video) return "A plain video can't be imported on this machine. Drop a Stray Scanner export's zip.";
  if (!file.size) return 'That file is empty.';
  if (file.size > maxMb * 2 ** 20) return `That file is larger than the ${maxMb} MB this takes.`;
  return null;
}

/** What a running import is doing, in a few words. */
export function stepText(job: WalkJob): string {
  if (job.status === 'uploading') return `Uploading ${job.name}`;
  if (job.status === 'extracting') return `Unpacking ${job.name}`;
  return job.detail ? `Importing ${job.name}: ${job.detail}` : `Importing ${job.name}`;
}

const MATCHED_BY: Record<string, string> = {
  map: "it fits that place's map",
  plan: 'it fits the plan you drew for that place',
  cameras: "that place's cameras see where it went",
};

/** What a finished import did: which place the walk went into and why it's that one, or that it made a new place.
 *  `name` is that place's name, once the places have been read again. */
export function resultText(job: WalkJob, name: string | undefined): string {
  if (job.result.new_place === 'True')
    return `${job.name} matched no place you already have, so it became a new place${name ? `: ${name}` : ''}.`;
  const why = MATCHED_BY[job.result.registered ?? ''];
  return `${job.name} went into ${name ?? 'a place you already have'}${why ? `: ${why}` : ''}.`;
}

const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer.types).includes('Files');

/** `names` are the listed places' names by id, and `onOpen` shows a place: the result names the walk's place and
 *  offers to open it. */
export function WalkImport({ names, onOpen }: { names: Map<string, string>; onOpen: (place: string) => void }) {
  const client = useQueryClient();
  const walk = useQuery(walkQuery());
  const [sent, setSent] = useState<{ name: string; share: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const resetFile = useRef<() => void>(null);
  const job = walk.data?.job ?? null;
  const video = walk.data?.video ?? false;
  const maxMb = walk.data?.max_mb ?? 4096;
  const busy = sent !== null || walkActive(job);
  const place = job?.result.final_place ?? '';
  const made = job?.result.new_place === 'True';

  // A finished import changed the places (a new revision in one, or a new place): read them again, once per job.
  const finished = job?.status === 'done' ? job.id : null;
  const reread = useRef<string | null>(null);
  useEffect(() => {
    if (!finished || reread.current === finished) return;
    reread.current = finished;
    void client.invalidateQueries({ queryKey: ['places'] });
  }, [finished, client]);

  const setJob = (next: WalkJob | null) =>
    client.setQueryData(walkQuery().queryKey, old => (old ? { ...old, job: next } : old));
  const send = async (file: File | null | undefined) => {
    if (!file || busy) return;
    const why = refusal(file, maxMb, video);
    setErr(why);
    if (why) return;
    setSent({ name: file.name, share: 0 });
    try {
      const reply = await uploadWalk(file, share => setSent({ name: file.name, share }));
      setJob(reply.job);
    } catch (e) {
      setErr(messageOf(e));
    } finally {
      setSent(null);
      void walk.refetch();
    }
  };
  const act = async (what: () => Promise<{ job: WalkJob | null }>) => {
    setErr(null);
    try {
      setJob((await what()).job);
    } catch (e) {
      setErr(messageOf(e));
    }
    void walk.refetch();
  };

  return (
    <section
      aria-label="Import a walk"
      className={classes.walkDrop}
      data-over={(over && !busy) || undefined}
      onDragEnter={e => hasFiles(e) && setOver(true)}
      onDragOver={e => {
        if (hasFiles(e)) e.preventDefault(); // without this the browser opens the file instead
      }}
      onDragLeave={e => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false);
      }}
      onDrop={e => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        setOver(false);
        void send(e.dataTransfer.files[0]);
      }}
    >
      <Text size="sm">
        Drop the zip Stray Scanner exports here to import a walk
        {video ? ', or a plain video of the walk (less accurate, and it marks no room as walked).' : '.'} It goes into
        the place it matches, or becomes a new place when it matches none.{' '}
        <Anchor href={WALKTHROUGH_DOC} target="_blank" rel="noreferrer">
          How to record a walk
        </Anchor>
      </Text>
      <Group mt={6} gap="xs">
        <FileButton
          accept={['.zip', ...(video ? VIDEO : [])].join(',')}
          inputProps={{ 'aria-label': 'Choose a walk to import' }}
          resetRef={resetFile}
          disabled={busy}
          onChange={file => {
            void send(file);
            resetFile.current?.();
          }}
        >
          {props => (
            <Button {...props} variant="default" size="xs" disabled={busy}>
              Choose file
            </Button>
          )}
        </FileButton>
      </Group>
      {sent && (
        <div role="status" className={classes.walkStatus}>
          <Text size="sm">
            Uploading {sent.name}: {Math.round(sent.share * 100)}%
          </Text>
          <Progress value={Math.round(sent.share * 100)} size="sm" aria-label="Uploaded so far" />
        </div>
      )}
      {!sent && job && walkActive(job) && (
        <div role="status" className={classes.walkStatus}>
          <Text size="sm">{stepText(job)}</Text>
          <Progress value={100} striped animated size="sm" aria-label="Importing" />
          <Text size="xs" c="dimmed">
            This takes a few minutes. It keeps going if you leave the page.
          </Text>
        </div>
      )}
      {!sent && job?.status === 'done' && (
        <Alert mt={8} color="green" title={made ? 'New place from a walk' : 'Walk imported'} role="status">
          <Text size="sm">{resultText(job, names.get(place))}</Text>
          <Group mt={6} gap="xs">
            {names.has(place) && (
              <Button size="xs" variant="default" onClick={() => onOpen(place)}>
                Open {names.get(place)}
              </Button>
            )}
            <Button size="xs" variant="default" onClick={() => void act(dismissWalk)}>
              Dismiss
            </Button>
          </Group>
        </Alert>
      )}
      {!sent && job?.status === 'failed' && (
        <Alert mt={8} color="red" title={`${job.name} wasn't imported`} role="alert">
          <Text size="sm">{job.error}</Text>
          <Group mt={6} gap="xs">
            {job.retry && (
              <Button size="xs" variant="default" onClick={() => void act(retryWalk)}>
                Try again
              </Button>
            )}
            <Button size="xs" variant="default" onClick={() => void act(dismissWalk)}>
              Dismiss
            </Button>
          </Group>
        </Alert>
      )}
      {err && (
        <Text size="sm" c="red" mt={6} role="alert">
          {err}
        </Text>
      )}
    </section>
  );
}
