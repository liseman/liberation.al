// The boxes over Review's large camera: who and what it saw at the playhead, drawn by the shared box layer
// (camera/BoxLayer, Live's style: named green with their name, unknown amber "Who's this?", things in their own
// colour). Naming freezes the displayed pixels and their actual timestamp, not the asynchronously moving playhead.
// A box matched to someone (a person or a pet) shows "Not <name>" when hovered or tapped: it unmatches that
// track through the corrections People and Pets already use, redraws the box unnamed at once, and offers Undo.
import { Button, Text } from '@mantine/core';
import { IconUserMinus } from '@tabler/icons-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BoxLayer } from '../camera/BoxLayer';
import type { BoxPick } from '../camera/boxes';
import { canvasFrameMetadata, sourceBoxToDecoded } from '../camera/geometry';
import type { Message } from '../lib/message';
import { Toast } from '../lib/Toast';
import { correct, undoChange } from '../people/api';
import { type LiveFaceDraft, prepareLiveFace } from '../people/liveFace';
import { markNot } from '../people/pets/api';
import { href, navigate } from '../router';
import { asBox, BUCKET_S, boxesAt, boxesQuery, faceToName, type ReviewBox, shownBoxes } from './boxes';
import { useEngine, useReview } from './context';
import classes from './player.module.css';

const RECENT_S = 15 * 60; // a minute this close to now may still be scanned, so it's read again after a while
const NEXT_S = 10; // this close to the end of a minute, the next one is fetched
/** A scrub crosses many minutes: the minute whose boxes are fetched moves only once the playhead holds this long, and
 *  a minute left before its boxes came is cancelled (one scrub fetched /review/boxes 16 times, the audit). */
export const SETTLE_MS = 150;

const HOVER_LEAVE_MS = 300; // the pointer can cross from a box to its "Not <name>" without the button going

/** A box matched to someone: a person's body or face, or a pet, with the track its match is on. */
type Matched = ReviewBox & { name: string; track: string };
/** The box as a matched one, or null for a box with no name or no track (an object, a stranger). */
function matched(row: ReviewBox | undefined): Matched | null {
  if (row?.tone !== 'known' || !row.name || !row.track) return null;
  if (row.kind === 'pet') return row.pet_id ? (row as Matched) : null;
  return row.kind === 'person' || row.kind === 'face' ? (row as Matched) : null;
}
/** What a box looks like once its match is taken back: a person nobody's named, or the animal by its species. */
const unnamed = (row: ReviewBox): ReviewBox => ({
  ...row,
  tone: 'unknown',
  name: undefined,
  pet_id: undefined,
  label: row.kind === 'pet' ? (row.species ?? 'animal') : "Who's this?",
});
const trackKey = (row: { kind: string; track?: string }) => `${row.kind === 'pet' ? 'pet' : 'person'}:${row.track}`;

/** value, once it has held for ms (at once the first time, and whenever it goes back to null). */
function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    if (value === settled) return;
    if (settled == null || value == null) {
      setSettled(value);
      return;
    }
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, settled, ms]);
  return settled;
}

export function ReviewBoxes({
  cam,
  getCanvas,
  onFace,
  kinds = ['people'],
  knownOnly = true,
}: {
  cam: string;
  getCanvas: () => HTMLCanvasElement | null;
  onFace: (draft: LiveFaceDraft) => void;
  kinds?: readonly ('people' | 'pets' | 'objects')[];
  knownOnly?: boolean; // Objects shows only remembered objects, not the camera's generic labels
}) {
  const engine = useEngine();
  const t = useReview(s => s.t);
  const sel = useReview(s => s.sel);
  const metadata = canvasFrameMetadata(getCanvas());
  const displayedAt = metadata ? metadata.capturedAtUs / 1e6 : null;
  const queryAt = displayedAt ?? t;
  const client = useQueryClient();
  // Tracks unmatched here, drawn unnamed straight away while the server's boxes catch up (and after: it agrees).
  const [unmatched, setUnmatched] = useState<ReadonlySet<string>>(() => new Set());
  const bucket = useSettled(queryAt == null ? null : Math.floor(queryAt / BUCKET_S), SETTLE_MS);
  useEffect(() => {
    if (bucket == null) return;
    // Leaving a minute before its boxes came: stop fetching them (they'd be drawn for nobody).
    return () => void client.cancelQueries({ queryKey: boxesQuery(cam, bucket, false).queryKey, exact: true });
  }, [client, cam, bucket]);
  const recent = bucket != null && Date.now() / 1000 - bucket * BUCKET_S < RECENT_S;
  const here = useQuery({ ...boxesQuery(cam, bucket ?? 0, recent), enabled: bucket != null });
  // Near the end of the settled minute, with the playhead in it (not mid-scrub elsewhere).
  const soon =
    queryAt != null &&
    bucket != null &&
    Math.floor(queryAt / BUCKET_S) === bucket &&
    (bucket + 1) * BUCKET_S - queryAt < NEXT_S;
  useQuery({ ...boxesQuery(cam, (bucket ?? 0) + 1, recent), enabled: soon }); // ready when playback gets there
  const shown = useMemo(() => {
    if (displayedAt == null || !here.data) return [];
    const enabled = here.data.filter(
      row =>
        kinds.includes(
          row.kind === 'face' || row.kind === 'person' ? 'people' : row.kind === 'pet' ? 'pets' : 'objects',
        ) && !(row.kind === 'label' && knownOnly),
    );
    return shownBoxes(boxesAt(enabled, displayedAt), engine.chips, sel).map(row =>
      row.track && unmatched.has(trackKey(row)) ? unnamed(row) : row,
    ); // sel changes with the chips too
  }, [here.data, displayedAt, sel, engine, kinds, knownOnly, unmatched]);
  const boxes = shown.flatMap(row => {
    const box = metadata && sourceBoxToDecoded(row.box, metadata);
    return box ? [{ ...asBox(row), box }] : [];
  });
  const [error, setError] = useState('');
  // The box whose "Not <name>" shows, and where; shown by hovering a box with a mouse, or by tapping it.
  const [menu, setMenu] = useState<{ row: Matched; pick: BoxPick } | null>(null);
  const leave = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const menuEl = useRef<HTMLDivElement>(null);
  const stay = () => clearTimeout(leave.current);
  const go = useCallback(() => {
    clearTimeout(leave.current);
    leave.current = setTimeout(() => setMenu(null), HOVER_LEAVE_MS);
  }, []);
  useEffect(() => () => clearTimeout(leave.current), []);
  const reveal = useCallback(
    (p: BoxPick) => {
      const row = matched(shown.find(r => r.id === p.id));
      if (!row) return false;
      clearTimeout(leave.current);
      setMenu({ row, pick: p });
      return true;
    },
    [shown],
  );
  const hover = useCallback((p: BoxPick | null) => (p ? void reveal(p) : go()), [reveal, go]);
  // A tapped-open "Not <name>" goes with a tap anywhere else.
  useEffect(() => {
    if (!menu) return;
    const away = (e: PointerEvent) => {
      const target = e.target as Element | null;
      if (!menuEl.current?.contains(target) && !target?.closest?.('.bx')) setMenu(null);
    };
    document.addEventListener('pointerdown', away, true);
    return () => document.removeEventListener('pointerdown', away, true);
  }, [menu]);
  const [said, setSaid] = useState<Message<'undo'>>(null);
  const undoing = useRef<(() => Promise<unknown>) | null>(null);
  const [busy, setBusy] = useState(false);
  const refetch = () => void client.invalidateQueries({ queryKey: ['review', 'boxes', cam] });
  const mark = (key: string, on: boolean) =>
    setUnmatched(prev => {
      const next = new Set(prev);
      if (on) next.add(key);
      else next.delete(key);
      return next;
    });
  const unmatch = async (row: Matched) => {
    setMenu(null);
    const key = trackKey(row);
    mark(key, true); // unnamed now, before the server says so
    try {
      let undo: () => Promise<unknown>;
      if (row.kind === 'pet' && row.pet_id) {
        const pet = row.pet_id;
        await markNot([row.track], pet, true);
        undo = () => markNot([row.track], pet, false);
      } else {
        const at = displayedAt ?? row.ts;
        const reply = await correct({
          name: row.name,
          kind: 'track',
          id: `${cam}:${row.track}@${at.toFixed(2)}`,
          action: 'remove',
        });
        const token = reply.undo;
        undo = () => (token ? undoChange(token) : Promise.resolve());
      }
      undoing.current = undo;
      setSaid({
        text: `Not ${row.name}: ${row.kind === 'pet' ? 'this animal' : 'this person'} is unnamed now.`,
        undo: 'undo',
      });
      refetch();
    } catch (reason) {
      mark(key, false);
      setSaid({ err: reason instanceof Error ? reason.message : `Could not take ${row.name} off this box.` });
    }
  };
  const undo = async () => {
    const run = undoing.current;
    if (!run) return;
    setBusy(true);
    try {
      await run();
      undoing.current = null;
      setUnmatched(new Set()); // the server's boxes say who it is again
      setSaid({ text: 'Put back.' });
      refetch();
    } catch (reason) {
      setSaid({ err: reason instanceof Error ? reason.message : 'Could not undo that.' });
    } finally {
      setBusy(false);
    }
  };
  const pick = useCallback(
    (p: BoxPick) => {
      if (reveal(p)) return; // a matched box: its "Not <name>", and the person's card beside it
      if (p.kind !== 'face') return;
      try {
        engine.pause();
        const canvas = getCanvas();
        const at = Number(canvas?.dataset.capturedAt);
        const frameMetadata = canvasFrameMetadata(canvas);
        if (!canvas || !frameMetadata || !Number.isFinite(at) || at <= 0)
          throw new Error('No displayed face picture is available.');
        // A body is named from its track's latest face; a face (older recordings) from its own.
        const selected = shown.find(row => row.id === p.id && (row.kind === 'face' || row.kind === 'person'));
        const found = selected?.track ? faceToName(here.data ?? [], selected.track, at) : null;
        const face = found && (selected?.kind === 'person' || found.group === selected?.group) ? found : null;
        if (!face || face.group == null || !face.track)
          throw new Error(
            selected?.kind === 'person'
              ? "This person's face isn't in view here. Select them where they face the camera."
              : 'The displayed face evidence is not available. Select the face again.',
          );
        const box = sourceBoxToDecoded(face.box, frameMetadata);
        if (!box) throw new Error('The displayed face geometry is unavailable.');
        onFace(
          prepareLiveFace({ ...p, box, faceSource: { camera: cam, at, track: face.track, group: face.group } }, canvas),
        );
        setError('');
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Could not prepare the selected face.');
      }
    },
    [shown, cam, engine, getCanvas, here.data, onFace, reveal],
  );
  // A paused video gets no new boxes, so they stay until the playhead moves (Live's fade after a second of silence).
  return (
    <>
      <BoxLayer
        boxes={boxes}
        ts={t}
        onPick={pick}
        onHover={hover}
        pickFaces="all"
        pickPets
        staleS={Number.POSITIVE_INFINITY}
      />
      {menu && (
        // Over the box's label: its own row when there's room above the box, else just inside its top.
        <div
          ref={menuEl}
          role="toolbar"
          aria-label={menu.row.name}
          className={classes.boxMenu}
          style={{ left: menu.pick.left - 2, top: menu.pick.top >= 26 ? menu.pick.top - 24 : menu.pick.top + 2 }}
          onPointerEnter={stay}
          onPointerLeave={e => (e.pointerType === 'mouse' ? go() : undefined)}
          onKeyDown={e => {
            if (e.key === 'Escape') {
              e.stopPropagation();
              setMenu(null);
            }
          }}
        >
          <Button
            size="compact-xs"
            color="dark"
            leftSection={<IconUserMinus size={13} aria-hidden />}
            onClick={() => void unmatch(menu.row)}
          >
            Not {menu.row.name}
          </Button>
          {menu.row.kind !== 'pet' && (
            <Button
              size="compact-xs"
              variant="white"
              color="dark"
              onClick={() => navigate(`${href('/people')}#card=${encodeURIComponent(menu.row.name)}`)}
            >
              Card
            </Button>
          )}
        </div>
      )}
      <Toast message={said} onUndo={() => void undo()} busy={busy} onClose={() => setSaid(null)} />
      {error && (
        <Text className={classes.faceError} role="alert">
          {error}
        </Text>
      )}
    </>
  );
}
