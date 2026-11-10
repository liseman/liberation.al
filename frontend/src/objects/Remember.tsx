import {
  Button,
  Card,
  Group,
  LoadingOverlay,
  NativeSelect,
  Stack,
  TagsInput,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import { useMutation } from '@tanstack/react-query';
import { type FormEvent, useCallback, useEffect, useEffectEvent, useRef, useState } from 'react';
import { LiveCam } from '../dashboard/camera/LiveCam';
import { Toast } from '../dashboard/lib/Toast';
import { cameraName } from '../dashboard/people/profile/role';
import { messageOf } from '../shared/http';
import {
  type Box,
  cancelSegment,
  enroll,
  evidenceUrl,
  type FoundObject,
  findObjects,
  getDraft,
  importView,
  type Segmentation,
  type Snapshot,
  segment,
} from './api';
import { FrameEditor } from './FrameEditor';
import { timestamp } from './format';
import { normalized } from './geometry';
import type { ObjectDraft } from './handoff';
import { loadImage } from './image';
import classes from './objects.module.css';

interface Frozen {
  snapshot: Snapshot;
  img: HTMLImageElement;
}
/** Find objects says so when a run takes longer than this. */
const SLOW_FIND_MS = 4000;
/** What EdgeTAM is asked to outline: a box or a point the owner chose, normalized to the frozen frame. */
type Prompt = { bbox: Box } | { point: [number, number] };

/** Freeze a camera view, then outline the object the owner points at with EdgeTAM (a clicked Live box, a click, or a
 *  drawn box), or ask EdgeTAM for every object in view (Find objects, #2020) and pick one, and save it. EdgeTAM runs only
 *  when the owner asks (#1481). */
export function Remember({
  cameras,
  cameraNames = {},
  onSaved,
  draft = null,
}: {
  cameras: string[];
  /** Each camera's name (vision.camera_names), by id: what the select and captions show. */
  cameraNames?: Record<string, string>;
  onSaved: () => void;
  draft?: ObjectDraft | null;
}) {
  const [camera, setCamera] = useState(cameras[0] ?? 'cam0');
  const [message, setMessage] = useState({ text: '', error: false });
  const [frozen, setFrozen] = useState<Frozen | null>(null);
  const [selection, setSelection] = useState<{ result: Segmentation; mask: HTMLImageElement } | null>(null);
  const [found, setFound] = useState<FoundObject[]>([]);
  const [crop, setCrop] = useState<Box | null>(null);
  const [directCrop, setDirectCrop] = useState<Box | null>(null);
  const [selectRequest, setSelectRequest] = useState(0);
  const [finding, setFinding] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const processing = preparing || finding;
  const liveCanvas = useRef<HTMLCanvasElement | null>(null);
  const revision = useRef(0);
  // Each pick among the found objects: a mask that loads after a newer pick is stale and never becomes the selection.
  const pickSeq = useRef(0);
  const request = useRef<AbortController | null>(null);
  const job = useRef<string | null>(null);
  const cancellation = useRef<Promise<unknown>>(Promise.resolve());
  const pendingCancellation = useRef<{ job: string; promise: Promise<unknown> } | null>(null);

  const cancelCurrent = useCallback(() => {
    request.current?.abort();
    request.current = null;
    const previous = job.current;
    if (!previous) {
      cancellation.current = Promise.resolve();
    } else if (pendingCancellation.current?.job === previous) {
      cancellation.current = pendingCancellation.current.promise;
    } else {
      const promise = cancelSegment(previous)
        .then(() => {
          if (job.current === previous) job.current = null;
        })
        .finally(() => {
          // Failed termination keeps job ownership, so the next action can retry it.
          if (pendingCancellation.current?.promise === promise) pendingCancellation.current = null;
        });
      pendingCancellation.current = { job: previous, promise };
      cancellation.current = promise;
    }
    return cancellation.current;
  }, []);
  const [name, setName] = useState('');
  const [aliases, setAliases] = useState<string[]>([]);
  const [aliasDraft, setAliasDraft] = useState('');
  const [note, setNote] = useState('');
  const say = (text: string, error = false) => setMessage({ text, error });
  const [saved, setSaved] = useState<string | null>(null); // the toast after remembering one (#1984)

  useEffect(
    () => () => {
      revision.current++;
      void cancelCurrent().catch(() => {});
      setFinding(false);
      setPreparing(false);
    },
    [cancelCurrent],
  );

  const reset = (clearCrop = true) => {
    revision.current++;
    void cancelCurrent().catch(() => {});
    setSelection(null);
    setDirectCrop(null);
    if (clearCrop) {
      setCrop(null);
      setFound([]);
    }
    setFinding(false);
    setPreparing(false);
    return revision.current;
  };

  const chooseAvailableCamera = useEffectEvent((next: string) => {
    reset();
    setFrozen(null);
    setCamera(next);
    say('');
  });
  useEffect(() => {
    const next = cameras[0];
    // Imported Live/Review frames may arrive before the authoritative status camera list.
    if (next && !frozen && !preparing && !cameras.includes(camera)) chooseAvailableCamera(next);
  }, [camera, cameras, frozen, preparing]);

  async function findObject(view: Frozen, expected: number, prompt: Prompt) {
    let ownedJob: string | null = null;
    pickSeq.current++; // a new outline or search: a pick from before it that's still loading is stale
    setSelectRequest(value => value + 1);
    setSelection(null);
    setDirectCrop(null);
    setFinding(true);
    say('');
    try {
      await cancellation.current;
      if (revision.current !== expected) return;
      const controller = new AbortController();
      request.current = controller;
      // getRandomValues works on local HTTP dashboards too; randomUUID requires a secure context.
      ownedJob = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte =>
        byte.toString(16).padStart(2, '0'),
      ).join('');
      job.current = ownedJob;
      const result = await segment(view.snapshot.token, prompt, ownedJob, controller.signal);
      if (job.current === ownedJob) job.current = null;
      if (revision.current !== expected) return;
      if (result.snapshot !== view.snapshot.token) throw new Error('The outline belongs to another camera view.');
      const mask = await loadImage(evidenceUrl(result.mask));
      if (revision.current !== expected) return;
      setSelection({ result, mask });
      say('Outlined. Name it, then remember it.');
    } catch (e) {
      if (revision.current !== expected) return;
      say(`Could not outline that: ${messageOf(e)}`, true);
    } finally {
      if (revision.current === expected) {
        setFinding(false);
        request.current = null;
      }
    }
  }

  /** Every object EdgeTAM finds in `region` (the part of the frame in view), shown as outlines to pick from. */
  async function findAll(view: Frozen, expected: number, region: Box) {
    let ownedJob: string | null = null;
    pickSeq.current++; // a new outline or search: a pick from before it that's still loading is stale
    setSelectRequest(value => value + 1);
    setSelection(null);
    setDirectCrop(null);
    setFound([]);
    setFinding(true);
    say('');
    const slow = window.setTimeout(() => {
      if (revision.current === expected)
        say('Still finding objects. A wide, busy view takes a few seconds; zooming in is faster to look through.');
    }, SLOW_FIND_MS);
    try {
      await cancellation.current;
      if (revision.current !== expected) return;
      const controller = new AbortController();
      request.current = controller;
      ownedJob = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte =>
        byte.toString(16).padStart(2, '0'),
      ).join('');
      job.current = ownedJob;
      const result = await findObjects(view.snapshot.token, region, ownedJob, controller.signal);
      if (job.current === ownedJob) job.current = null;
      if (revision.current !== expected) return;
      if (result.snapshot !== view.snapshot.token) throw new Error('The objects belong to another camera view.');
      setFound(result.objects);
      const seconds = (result.ms / 1000).toFixed(1);
      say(
        result.objects.length
          ? `Found ${result.objects.length} object${result.objects.length === 1 ? '' : 's'} in ${seconds} s. Click one to select it.`
          : 'No objects found in view. Zoom in, or click an object to outline it.',
      );
    } catch (e) {
      if (revision.current !== expected) return;
      say(`Could not find objects: ${messageOf(e)}`, true);
    } finally {
      window.clearTimeout(slow);
      if (revision.current === expected) {
        setFinding(false);
        request.current = null;
      }
    }
  }

  async function choose(id: string) {
    const picked = found.find(f => f.selection === id);
    if (!picked || !frozen || finding) return;
    const expected = revision.current;
    const mine = ++pickSeq.current;
    setDirectCrop(null);
    setSelection(null); // nothing is saveable until this pick's mask has loaded
    try {
      const mask = await loadImage(evidenceUrl(picked.mask));
      if (revision.current !== expected || pickSeq.current !== mine) return;
      setSelection({ result: picked, mask });
      say('Selected. Name it, then remember it.');
    } catch (e) {
      if (revision.current === expected && pickSeq.current === mine) say(messageOf(e), true);
    }
  }

  async function show(snapshot: Snapshot, draftName: string, expected: number, box?: Box) {
    const img = await loadImage(evidenceUrl(snapshot.evidence));
    if (revision.current !== expected) return;
    const view = { snapshot, img };
    setFrozen(view);
    setCamera(snapshot.camera);
    setName(draftName);
    setPreparing(false);
    // A box clicked on Live is outlined once; any other frozen view waits for a click or a drawn box.
    if (box) void findObject(view, expected, { bbox: box });
    else say('Click the object, or draw a box around it, to outline it.');
  }

  const freeze = useMutation({
    mutationFn: async ({ expected, picture }: { expected: number; picture: ObjectDraft }) => {
      await cancellation.current;
      if (revision.current === expected) await show(await importView(picture), '', expected);
    },
    onError: (e, { expected }) => {
      if (revision.current === expected) {
        setPreparing(false);
        say(messageOf(e), true);
      }
    },
  });
  const loadDraft = useMutation({
    mutationFn: async (expected: number) => {
      await cancellation.current;
      if (revision.current !== expected) return;
      const { draft } = await getDraft();
      if (!draft)
        throw new Error('Nothing said to remember yet. Say “remember this as my SSD”, or freeze a view here.');
      await show(draft.snapshot, draft.name, expected);
    },
    onError: (e, expected) => {
      if (revision.current === expected) {
        setPreparing(false);
        say(messageOf(e), true);
      }
    },
  });
  const importedDraft = useRef<ObjectDraft | null>(null);
  const loadIncomingDraft = useEffectEvent(async (draft: ObjectDraft) => {
    if (save.isPending) return;
    const expected = reset();
    setPreparing(true);
    try {
      await cancellation.current;
      if (revision.current === expected) {
        await show(
          await importView(draft),
          draft.name ?? '',
          expected,
          draft.kind === 'full_frame' ? undefined : draft.box,
        );
        if (revision.current === expected) importedDraft.current = draft;
      }
    } catch (error) {
      if (revision.current === expected) {
        setPreparing(false);
        say(messageOf(error), true);
      }
    }
  });
  const save = useMutation({
    mutationFn: (args: { snapshot: Snapshot; selection: Segmentation | null; crop: Box | null; bbox: Box }) =>
      enroll({
        name,
        camera: args.snapshot.camera,
        snapshot: args.snapshot.token,
        ...(args.selection
          ? { selection: args.selection.selection }
          : args.crop
            ? { crop: args.crop, direct_crop: true }
            : { full_crop: true }),
        bbox: args.bbox,
        aliases: [...aliases, aliasDraft.trim()].filter(Boolean).join(', '),
        note,
      }),
    onSuccess: d => {
      reset();
      say('');
      setSaved(
        d.object.mode === 'visual'
          ? 'Remembered. Looking for conservative visual matches in future frames.'
          : 'Remembered as a manual sighting. This crop lacks enough visual detail for automatic matching.',
      );
      setFrozen(null);
      onSaved();
    },
    onError: e => say(messageOf(e), true),
  });
  useEffect(() => {
    if (save.isPending || !draft || importedDraft.current === draft) return;
    void loadIncomingDraft(draft);
  }, [draft, save.isPending]);

  const pick = (x: number, y: number) => {
    if (finding || !frozen || save.isPending) return;
    const point: [number, number] = [x / frozen.img.naturalWidth, y / frozen.img.naturalHeight];
    void findObject(frozen, reset(false), { point });
  };
  const changeCrop = (box: Box | null) => {
    // Editing the previous frame must not invalidate its pending replacement.
    if (save.isPending || preparing) return;
    reset(false);
    setCrop(box);
    say('');
  };
  const fullCrop = frozen?.snapshot.source === 'live_crop';
  const liveFallback = fullCrop && !crop;
  const canRemember =
    !!frozen &&
    !preparing &&
    (!!selection || !!directCrop || liveFallback) &&
    (!finding || liveFallback || !!directCrop);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!canRemember || !frozen || save.isPending) return;
    // A Live crop can be remembered before SAM finishes. Stop accepting its results immediately,
    // and cancel its owned child without delaying the explicit save on model termination.
    revision.current++;
    void cancelCurrent().catch(() => {});
    setFinding(false);
    setPreparing(false);
    const chosen = directCrop ? ([...directCrop] as Box) : null;
    save.mutate({
      snapshot: frozen.snapshot,
      selection: selection?.result ?? null,
      crop: chosen,
      bbox:
        selection?.result.bbox ??
        (chosen
          ? normalized(chosen, { width: frozen.img.naturalWidth, height: frozen.img.naturalHeight })
          : [0, 0, 1, 1]),
    });
  };

  return (
    <Card withBorder padding="md" radius="lg" component="section" className={classes.remember}>
      <Toast text={saved} onClose={() => setSaved(null)} />
      <Title order={2} mb="xs">
        Remember an object
      </Title>
      <Group mt="md" gap="sm">
        <NativeSelect
          aria-label="Camera"
          disabled={save.isPending}
          value={camera}
          onChange={e => {
            reset();
            setFrozen(null);
            setCamera(e.currentTarget.value);
            say('');
          }}
          data={cameras.map(c => ({ value: c, label: cameraName(c, cameraNames[c]) }))}
        />
        <Button
          variant="default"
          onClick={() => {
            const canvas = liveCanvas.current;
            const capturedAt = Number(canvas?.dataset.capturedAt);
            if (!canvas || !Number.isFinite(capturedAt) || capturedAt <= 0) {
              say('Wait for a picture from this camera before freezing a view.', true);
              return;
            }
            try {
              // Copy the displayed pixels now, before waiting for an older EdgeTAM job to stop.
              const picture: ObjectDraft = {
                kind: 'full_frame',
                camera,
                image: canvas.toDataURL('image/jpeg', 0.95),
                captured_at: capturedAt,
              };
              const expected = reset();
              setPreparing(true);
              say('');
              freeze.mutate({ expected, picture });
            } catch (e) {
              say(messageOf(e), true);
            }
          }}
          loading={freeze.isPending}
          disabled={save.isPending}
        >
          Freeze view
        </Button>
        <Button
          variant="default"
          onClick={() => {
            const expected = reset();
            setPreparing(true);
            say('');
            loadDraft.mutate(expected);
          }}
          loading={loadDraft.isPending}
          disabled={save.isPending}
        >
          Use what I said
        </Button>
      </Group>
      <div className={classes.livePreview} hidden={!!frozen} style={{ marginTop: 12 }}>
        <LiveCam key={camera} cam={camera} alt={`Live view from ${camera}`} canvasRef={liveCanvas} />
        <span className={classes.cameraLabel}>{cameraName(camera, cameraNames[camera])}</span>
      </div>
      {!frozen && (
        <Text size="sm" c="dimmed" mt="xs">
          Live camera preview. Freeze the view, then click an object or draw a box around it.
        </Text>
      )}
      {!frozen && message.text && (
        <Text role={message.error ? 'alert' : 'status'} mt="sm" c={message.error ? 'var(--lib-bad)' : undefined}>
          {message.text}
        </Text>
      )}
      {(frozen || processing) && (
        <Stack gap="xs" mt="sm">
          {frozen && (
            <Text
              c="dimmed"
              size="sm"
            >{`${cameraName(frozen.snapshot.camera, cameraNames[frozen.snapshot.camera])} · captured ${timestamp(frozen.snapshot.captured_at)} · frozen, not live`}</Text>
          )}
          <div className={classes.preview} aria-busy={processing}>
            {frozen ? (
              <FrameEditor
                key={frozen.snapshot.token}
                image={frozen.img}
                overlay={selection?.mask}
                selectionBox={selection?.result.bbox}
                crop={crop}
                onCropChange={changeCrop}
                onPick={pick}
                busy={processing}
                disabled={save.isPending || preparing}
                selectRequest={selectRequest}
                onFindObjects={region => void findAll(frozen, reset(false), region)}
                found={found.map(f => ({ id: f.selection, outline: f.outline }))}
                chosen={selection?.result.selection ?? null}
                onChoose={id => void choose(id)}
              />
            ) : (
              <div className={classes.waitingFrame} />
            )}
            <LoadingOverlay
              visible={processing}
              role="status"
              aria-label="Processing this view"
              className={classes.processing}
              overlayProps={{ backgroundOpacity: 0 }}
              loaderProps={{ size: 'md', 'aria-hidden': true }}
              zIndex={1}
            />
          </div>
          {frozen && message.text && (
            <Text role={message.error ? 'alert' : 'status'} c={message.error ? 'var(--lib-bad)' : undefined}>
              {message.text}
            </Text>
          )}
          {frozen && (
            <>
              {crop && (
                <>
                  <Text size="sm">{`Crop: ${crop[2]} × ${crop[3]} pixels`}</Text>
                  <Group gap="sm">
                    <Button
                      variant="default"
                      disabled={crop[2] < 24 || crop[3] < 24 || save.isPending}
                      onClick={() => {
                        const chosen = [...crop] as Box;
                        reset(false);
                        setDirectCrop(chosen);
                        say('Crop selected. Name it, then remember it.');
                      }}
                    >
                      Use this crop
                    </Button>
                    <Button
                      variant="default"
                      disabled={save.isPending || preparing}
                      onClick={() => {
                        const size = { width: frozen.img.naturalWidth, height: frozen.img.naturalHeight };
                        const bbox = normalized([...crop] as Box, size);
                        void findObject(frozen, reset(false), { bbox });
                      }}
                    >
                      Find object in box
                    </Button>
                  </Group>
                  {(crop[2] < 24 || crop[3] < 24) && (
                    <Text size="xs" c="dimmed">
                      Choose at least 24 × 24 pixels to remember a crop directly.
                    </Text>
                  )}
                </>
              )}
              <form onSubmit={submit}>
                <Stack gap="sm">
                  <TextInput
                    label="Object name"
                    required
                    maxLength={80}
                    placeholder="my SSD"
                    value={name}
                    onChange={e => setName(e.currentTarget.value)}
                  />
                  <TagsInput
                    label="Other names (comma separated)"
                    maxLength={1000}
                    placeholder="drive, external SSD"
                    value={aliases}
                    onChange={setAliases}
                    searchValue={aliasDraft}
                    onSearchChange={setAliasDraft}
                  />
                  <Textarea
                    label="Note"
                    maxLength={2000}
                    rows={2}
                    placeholder="Silver drive with the orange sticker"
                    value={note}
                    onChange={e => setNote(e.currentTarget.value)}
                  />
                  <Group>
                    <Button type="submit" disabled={!canRemember} loading={save.isPending}>
                      {directCrop
                        ? 'Remember crop'
                        : liveFallback && !selection
                          ? 'Remember object'
                          : 'Remember selected object'}
                    </Button>
                  </Group>
                </Stack>
              </form>
            </>
          )}
        </Stack>
      )}
    </Card>
  );
}
