// Every picture of an unnamed face group, a page at a time, to pick the ones that aren't this person and take them out
// of it. Clicking a picture selects it (a ring and a check); "Not this person" removes the selected ones from
// the group for good, so naming it later learns only from what's left.
import { Button, Group, Loader, Text } from '@mantine/core';
import { IconUserMinus } from '@tabler/icons-react';
import { useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { ProcessingImages } from '../../shared/ProcessingImages';
import { SafeImage } from '../../shared/SafeImage';
import { plural } from '../lib/format';
import { SelectableGrid } from '../lib/SelectableGrid';
import { groupShots, rejectShots, type Shot } from './api';
import classes from './whosthis.module.css';

export const PAGE = 60;

export function ShotPicker({
  id,
  label,
  total,
  onRejected,
  onClose,
}: {
  id: number;
  label: string; // "Face 3"
  total: number; // how many pictures the row said it has
  onRejected: (result: { text: string } | { err: string }, emptied: boolean) => void;
  onClose: () => void;
}) {
  const [shots, setShots] = useState<Shot[]>([]);
  const [count, setCount] = useState(total);
  const [pending, setPending] = useState(0);
  const [unavailable, setUnavailable] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<ReadonlySet<string>>(() => new Set());
  const [saving, setSaving] = useState(false);

  const load = async (offset: number, signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const page = await groupShots(id, offset, PAGE, signal);
      if (signal?.aborted) return;
      setShots(have =>
        offset ? [...have, ...page.sightings.filter(s => !have.some(h => h.id === s.id))] : page.sightings,
      );
      setCount(page.total);
      setPending(page.shots_pending ?? 0);
      setUnavailable(page.shots_unavailable ?? 0);
    } catch (e) {
      if (!signal?.aborted) setError(`Could not load the pictures. ${messageOf(e)}`);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  };
  // biome-ignore lint/correctness/useExhaustiveDependencies: the first page, once per group
  useEffect(() => {
    const abort = new AbortController();
    void load(0, abort.signal);
    return () => abort.abort();
  }, [id]);

  const shown = shots.length;
  useEffect(() => {
    if (!pending) return;
    const abort = new AbortController();
    let refreshing = false;
    const timer = setInterval(async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        // Ready pictures may have appeared before any loaded page. Refresh the prefix rather than append shifted pages.
        const pages = [];
        for (let offset = 0; offset < Math.max(PAGE, shown); offset += PAGE)
          pages.push(await groupShots(id, offset, PAGE, abort.signal));
        if (abort.signal.aborted) return;
        setShots([...new Map(pages.flatMap(page => page.sightings).map(s => [s.id, s] as const)).values()]);
        setCount(pages[0]?.total ?? 0);
        setPending(pages[0]?.shots_pending ?? 0);
        setUnavailable(pages[0]?.shots_unavailable ?? 0);
        setError(null);
      } catch (e) {
        if (!abort.signal.aborted) setError(`Could not load the pictures. ${messageOf(e)}`);
      } finally {
        refreshing = false;
      }
    }, 2000);
    return () => {
      clearInterval(timer);
      abort.abort();
    };
  }, [id, pending, shown]);

  const toggle = (shot: string) =>
    setPicked(have => {
      const next = new Set(have);
      if (!next.delete(shot)) next.add(shot);
      return next;
    });
  const reject = async () => {
    if (!picked.size || saving) return;
    setSaving(true);
    try {
      const r = await rejectShots(id, [...picked]);
      if (!r.ok) throw new Error(r.error || 'The server refused that.');
      setShots(have => have.filter(s => !picked.has(s.id)));
      setCount(have => Math.max(0, have - r.removed));
      setPicked(new Set());
      const took = `Took ${plural(r.removed, 'picture', 'pictures')} out of ${label}.`;
      onRejected({ text: r.ignored ? `${took} Nothing was left, so it was put aside.` : took }, r.ignored);
    } catch (e) {
      onRejected({ err: `Could not take those out: ${messageOf(e)}` }, false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className={classes.picker} aria-label={`Every picture of ${label}`}>
      <Group justify="space-between" gap="xs">
        <Text size="xs" c="dimmed">
          {picked.size
            ? `${picked.size} of ${count} selected`
            : `${plural(count, 'picture', 'pictures')}. Select the ones that aren't this person.`}
        </Text>
        <Group gap={6}>
          <Button
            leftSection={<IconUserMinus size={14} aria-hidden />}
            size="compact-sm"
            variant="subtle"
            disabled={!picked.size || saving}
            onClick={reject}
          >
            Not this person
          </Button>
          <Button size="compact-sm" variant="subtle" onClick={onClose}>
            Hide
          </Button>
        </Group>
      </Group>
      <SelectableGrid
        label={`Pick the pictures of ${label} that aren't this person`}
        tiles={shots.map(s => ({
          key: s.id,
          label: `Picture from ${new Date(s.ts * 1000).toLocaleString()}`,
          content: <SafeImage className={classes.square} src={s.url} alt="" loading="lazy" />,
        }))}
        isOn={key => picked.has(key)}
        onToggle={toggle}
        tone="remove"
        disabled={saving}
      />
      <ProcessingImages count={pending} unavailable={unavailable} />
      {error && (
        <Text size="xs" c="var(--lib-bad)" role="alert">
          {error}
        </Text>
      )}
      {loading ? (
        <Loader size="xs" />
      ) : (
        shots.length < count && (
          <Button size="compact-xs" variant="default" onClick={() => void load(shots.length)}>
            Load {Math.min(PAGE, count - shots.length)} more
          </Button>
        )
      )}
    </section>
  );
}
