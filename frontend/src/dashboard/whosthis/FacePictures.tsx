// Every picture of an unnamed face group, a page at a time, for a selected Who's this? card: all start selected,
// and clicking one deselects it (naming/Pictures). The card's row keeps which ones are deselected and sends them
// with its answer, so a page not loaded yet counts as selected.
import { Button, Loader, Text } from '@mantine/core';
import { useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { ProcessingImages } from '../../shared/ProcessingImages';
import { Pictures } from '../naming/Pictures';
import { groupShots, type Shot } from './api';

export const PAGE = 60;

export function FacePictures({
  id,
  label,
  total,
  out,
  onToggle,
  onCount,
  disabled = false,
}: {
  id: number;
  label: string; // "Face 3"
  total: number; // how many pictures the row said it has
  out: ReadonlySet<string>; // the deselected ones
  onToggle: (shot: string) => void;
  onCount?: (count: number) => void; // how many pictures there are, once a page says
  disabled?: boolean;
}) {
  const [shots, setShots] = useState<Shot[]>([]);
  const [count, setCount] = useState(total);
  const [pending, setPending] = useState(0);
  const [unavailable, setUnavailable] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const counted = (n: number) => {
    setCount(n);
    onCount?.(n);
  };
  const load = async (offset: number, signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const page = await groupShots(id, offset, PAGE, signal);
      if (signal?.aborted) return;
      setShots(have =>
        offset ? [...have, ...page.sightings.filter(s => !have.some(h => h.id === s.id))] : page.sightings,
      );
      counted(page.total);
      setPending(page.shots_pending ?? 0);
      setUnavailable(page.shots_unavailable ?? 0);
    } catch (e) {
      if (!signal?.aborted) setError(`Could not load the pictures. ${messageOf(e)}`);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  };
  // biome-ignore lint/correctness/useExhaustiveDependencies: the first page, once per group and per change of its pictures
  useEffect(() => {
    const abort = new AbortController();
    void load(0, abort.signal);
    return () => abort.abort();
  }, [id, total]);

  const shown = shots.length;
  // biome-ignore lint/correctness/useExhaustiveDependencies: polls while pictures are still being made
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
        counted(pages[0]?.total ?? 0);
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

  return (
    <Pictures
      label={label}
      pictures={shots.map(s => ({
        key: s.id,
        label: `Picture from ${new Date(s.ts * 1000).toLocaleString()}`,
        src: s.url,
      }))}
      total={Math.max(count, shots.length)}
      out={out}
      onToggle={onToggle}
      disabled={disabled}
    >
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
    </Pictures>
  );
}
