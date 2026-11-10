// A camera's controls panel, loaded when its popover first opens (CameraControls.tsx): its sliders and form stay
// out of Live's first load.
import { Button, Checkbox, NativeSelect, Slider, Stack, Switch, Text } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { messageOf, request } from '../../shared/http';

export interface CameraControl {
  id: number;
  name: string;
  type: 'integer' | 'boolean' | 'menu';
  min: number;
  max: number;
  step: number;
  default: number;
  current: number;
  disabled: boolean;
  options?: { value: number; label: string }[];
}
export interface CameraControlsView {
  camera: string;
  available: boolean;
  pending?: boolean;
  note?: string;
  can_control: boolean;
  auto_exposure: boolean;
  controls: CameraControl[];
}
type Changes = { values?: Record<string, number | boolean>; auto_exposure?: boolean; reset?: boolean };

/** Each open panel owns one serial write queue; camera changes discard its pending work. */
export function CameraControlsPanel({ camera }: { camera: string }) {
  const [view, setView] = useState<CameraControlsView | null>(null);
  const [error, setError] = useState('');
  const [stalled, setStalled] = useState(false);
  const [pollRetry, setPollRetry] = useState(0);
  const queue = useRef<Changes[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const writing = useRef(false);
  const alive = useRef(true);
  const abort = useRef(new AbortController());
  const [reload, setReload] = useState(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: reload explicitly retries failed reads.
  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    abort.current = controller;
    request<CameraControlsView>(
      `/devices/camera-controls?camera=${encodeURIComponent(camera)}`,
      undefined,
      controller.signal,
    )
      .then(result => {
        if (!controller.signal.aborted) {
          setView(result);
          setError('');
        }
      })
      .catch(e => {
        if (!controller.signal.aborted) setError(messageOf(e));
      });
    return () => {
      alive.current = false;
      controller.abort();
      clearTimeout(timer.current);
      queue.current = [];
    };
  }, [camera, reload]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: pollRetry explicitly restarts a bounded polling attempt.
  useEffect(() => {
    if (!view?.pending) return;
    setStalled(false);
    let attempts = 0;
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout>;
    async function poll() {
      if (++attempts > 6) {
        setStalled(true);
        return;
      }
      try {
        if (!writing.current && queue.current.length === 0) {
          const result = await request<CameraControlsView>(
            `/devices/camera-controls?camera=${encodeURIComponent(camera)}`,
            undefined,
            controller.signal,
          );
          if (!controller.signal.aborted && !writing.current && queue.current.length === 0) {
            setView(old =>
              result.pending && old
                ? {
                    ...result,
                    controls: result.controls.map(control => ({
                      ...control,
                      current: old.controls.find(previous => previous.id === control.id)?.current ?? control.current,
                    })),
                    auto_exposure: old.auto_exposure,
                  }
                : result,
            );
            setError('');
            if (!result.pending) return;
          }
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(messageOf(e));
      }
      if (!controller.signal.aborted)
        timeout = setTimeout(() => {
          void poll();
        }, 1200);
    }
    timeout = setTimeout(() => {
      void poll();
    }, 1200);
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [camera, view?.pending, pollRetry]);

  async function flush() {
    if (writing.current || !alive.current || queue.current.length === 0) return;
    const changes = queue.current.shift();
    if (!changes) return;
    writing.current = true;
    try {
      const result = await request<CameraControlsView>(
        '/devices/camera-controls',
        { camera, ...changes },
        abort.current.signal,
      );
      if (alive.current) {
        // Keep values already edited while this write was in flight visible.
        let pending: Changes = result.pending ? { ...changes, values: { ...changes.values } } : {};
        for (const next of queue.current) {
          if (next.reset) pending = {};
          else {
            pending.values ??= {};
            Object.assign(pending.values, next.values);
            if (next.auto_exposure !== undefined) pending.auto_exposure = next.auto_exposure;
          }
        }
        setView({
          ...result,
          auto_exposure: pending.auto_exposure ?? result.auto_exposure,
          controls: result.controls.map(control => ({
            ...control,
            current: Number(pending.values?.[control.id] ?? control.current),
          })),
        });
        setError('');
        setStalled(false);
        if (result.pending) setPollRetry(value => value + 1);
      }
    } catch (e) {
      if (alive.current) setError(messageOf(e));
    } finally {
      writing.current = false;
      if (alive.current) {
        void flush();
      }
    }
  }
  function change(changes: Changes) {
    if (changes.reset) {
      // Reset is a barrier: writes on either side remain separate requests.
      queue.current.push(changes);
    } else {
      const last = queue.current.at(-1);
      if (last && !last.reset) {
        queue.current[queue.current.length - 1] = {
          ...last,
          ...changes,
          values: { ...last.values, ...changes.values },
        };
      } else {
        queue.current.push(changes);
      }
    }
    setView(
      old =>
        old && {
          ...old,
          auto_exposure: changes.auto_exposure ?? old.auto_exposure,
          controls: old.controls.map(control => ({
            ...control,
            current: Number(changes.values?.[control.id] ?? control.current),
          })),
        },
    );
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      void flush();
    }, 300);
  }
  return (
    <Stack gap="sm">
      <Text fw={600}>Camera controls</Text>
      {error && (
        <Text role="alert" c="red" size="sm">
          {error}
        </Text>
      )}
      {!view && !error && <Text size="sm">Loading controls…</Text>}
      {error && (
        <Button size="xs" variant="default" onClick={() => setReload(n => n + 1)}>
          Refresh controls
        </Button>
      )}
      {view?.pending && !stalled && (
        <Text size="sm" role="status">
          Saved. Applying camera controls…
        </Text>
      )}
      {stalled && (
        <>
          <Text size="sm" role="alert">
            Saved, but camera has not applied these values.
          </Text>
          <Button size="xs" variant="default" onClick={() => setPollRetry(value => value + 1)}>
            Check again
          </Button>
        </>
      )}
      {view?.note && <Text size="sm">{view.note}</Text>}
      {view?.available && (
        <>
          {!view.can_control && <Text size="sm">Camera controls are read only.</Text>}
          <Checkbox
            label="Automatic exposure"
            checked={view.auto_exposure}
            disabled={!view.can_control}
            onChange={event => change({ auto_exposure: event.currentTarget.checked })}
          />
          {view.controls.map(control => {
            const disabled = !view.can_control || control.disabled;
            const set = (value: number | boolean) => change({ values: { [control.id]: value } });
            if (control.type === 'boolean')
              return (
                <Switch
                  key={control.id}
                  label={control.name}
                  checked={Boolean(control.current)}
                  disabled={disabled}
                  onChange={event => set(event.currentTarget.checked)}
                />
              );
            if (control.type === 'menu')
              return (
                <NativeSelect
                  key={control.id}
                  label={control.name}
                  value={String(control.current)}
                  disabled={disabled}
                  data={(control.options ?? []).map(option => ({ value: String(option.value), label: option.label }))}
                  onChange={event => set(Number(event.currentTarget.value))}
                />
              );
            return (
              <div key={control.id}>
                <Text size="sm" id={`camera-control-${control.id}`}>
                  {control.name}: {control.current}
                </Text>
                <Slider
                  aria-labelledby={`camera-control-${control.id}`}
                  value={control.current}
                  min={control.min}
                  max={control.max}
                  step={control.step || 1}
                  disabled={disabled}
                  onChange={set}
                />
              </div>
            );
          })}
          <Text size="xs" c="dimmed">
            Reset saves currently adjustable controls to camera defaults. Automatic exposure is unchanged.
          </Text>
          <Button variant="default" size="xs" disabled={!view.can_control} onClick={() => change({ reset: true })}>
            Reset defaults
          </Button>
        </>
      )}
    </Stack>
  );
}
