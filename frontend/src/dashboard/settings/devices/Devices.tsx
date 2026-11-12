// Settings' Devices section: each camera with its live
// preview, its name, how it is doing, whether it is used, and Mirror; each microphone with its live level against the
// threshold speech has to pass, its sensitivity, and whether to listen with it. The names and choice of
// cameras and microphones are settings (Settings' draft and save bar; saving restarts what reads them). Sensitivity
// applies at once (POST /devices/sensitivity), and Mirror is this browser's own choice. Between the cameras
// and the microphones, the Network cameras card lists the ingest sources that are cameras. A camera or
// microphone that's off shows only its name and its checkbox; its settings come back when it's checked (SourceItem).
import { Anchor, Button, Checkbox, NativeSelect, Slider, Stack, Switch, Text } from '@mantine/core';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type CSSProperties, useEffect, useId, useRef, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { mirrored, setMirrored, usePrefs } from '../../lib/prefs';
import { failed, SAVE_IDLE, SaveState, type SaveStatus, saved, saving } from '../../lib/SaveState';
import { useLive } from '../../live/hooks';
import type { AmbientDevice } from '../../live/types';
import { micName } from '../../livetab/Mics';
import { viewName } from '../../review/views';
import { href, linkClick } from '../../router';
import { shell } from '../../shell';
import {
  type CameraRow,
  type CameraVideoChange,
  DEVICES_KEY,
  type Devices as DeviceView,
  gainText,
  getDevices,
  type MicGain,
  type MicRow,
  restartCapture,
  saveAutoGain,
  saveCamera,
  saveCameraVideo,
  saveDefaultSensitivity,
  saveMic,
  saveSensitivity,
} from '../../sources/api';
import { levelWidth, useLiveLevels } from '../../sources/levels';
import { ZoneEditor } from '../../zones/ZoneEditor';
import type { ControlProps } from '../controls';
import { useDraft } from '../draft';
import { SettingsRow, SettingsSection, SourceItem } from '../layout';
import layout from '../layout.module.css';
import { CONFIG_KEY } from '../model';
import { SettingRow } from '../Row';
import fields from '../settings.module.css';
import { CARD_ID } from '../sources/model';
import { NetworkCameras } from '../sources/Sources';
import { CameraHead, CameraName, toggleClasses } from './CameraCard';
import { DEVICE_CARD } from './cards';
import { deviceName } from './deviceName';
import classes from './devices.module.css';
import { useCameraStates } from './issues';
import { cameraStatus, parsePlaces, withPlace } from './model';

const useDeviceView = () =>
  useQuery({ queryKey: DEVICES_KEY, queryFn: getDevices, refetchOnWindowFocus: false, staleTime: 30_000 });

function BrightnessTarget({
  camera,
  off,
  pending,
  save,
  status,
}: {
  camera: CameraRow;
  off: boolean;
  pending: boolean;
  save: (change: CameraVideoChange) => void;
  status: SaveStatus;
}) {
  const value = camera.video?.exposure_target ?? 110;
  const [shown, setShown] = useState(value);
  useEffect(() => {
    if (!pending) setShown(value);
  }, [value, pending]);
  return (
    <SettingsRow className={fields.row}>
      <span className={layout.label}>Target brightness: {shown}</span>
      <div className={classes.markRow}>
        <Slider
          min={60}
          max={180}
          step={5}
          value={shown}
          onChange={setShown}
          onChangeEnd={target => save({ exposure_target: target })}
          disabled={off || !camera.video?.auto_exposure}
          thumbLabel={`Target brightness for ${deviceName(camera.name)}`}
        />
        <SaveState status={status} />
      </div>
    </SettingsRow>
  );
}

function CameraSettings({
  camera,
  off,
  pending,
  save,
  statusFor,
}: {
  camera: CameraRow;
  off: boolean;
  pending: boolean;
  save: (change: CameraVideoChange) => void;
  statusFor: (field: string) => SaveStatus; // this camera's save state, for the control the field names
}) {
  const resolutionId = useId();
  const exposureId = useId();
  const video = camera.video;
  if (!video) return null;
  const modes = (camera.modes?.choices ?? []).flatMap(size =>
    size.fps.map(fps => ({
      value: `${size.width}x${size.height}@${fps}`,
      label: `${size.width}×${size.height} at ${fps} fps`,
      width: size.width,
      height: size.height,
      fps,
    })),
  );
  const current = `${video.width}x${video.height}@${video.fps}`;
  const currentText = `${video.width}×${video.height} at ${video.fps} fps`;
  const data: { value: string; label: string; disabled?: boolean }[] = modes.map(({ value, label }) => ({
    value,
    label,
  }));
  if (!modes.some(mode => mode.value === current))
    data.unshift({ value: current, label: `${currentText} (current)`, disabled: true });
  const disabled = off || pending || !camera.selected || !camera.present;
  return (
    <>
      <SettingsRow className={fields.row}>
        <div className={layout.words}>
          {modes.length > 1 ? (
            <label htmlFor={resolutionId} className={layout.label}>
              Resolution
            </label>
          ) : (
            <span className={layout.label}>Resolution</span>
          )}
          <div id={`${resolutionId}-help`} className={layout.description}>
            Saving restarts only this camera.
            {!modes.length && camera.modes?.diagnostics[0] && <div>{camera.modes.diagnostics[0]}</div>}
          </div>
        </div>
        {modes.length > 1 ? (
          <div className={classes.markRow} data-align="end">
            <NativeSelect
              id={resolutionId}
              aria-label={`Resolution for ${deviceName(camera.name)}`}
              aria-describedby={`${resolutionId}-help`}
              value={current}
              data={data}
              disabled={disabled}
              onChange={event => {
                const mode = modes.find(item => item.value === event.currentTarget.value);
                if (mode) save({ width: mode.width, height: mode.height, fps: mode.fps });
              }}
            />
            <SaveState status={statusFor('resolution')} />
          </div>
        ) : (
          <Text size="sm">Resolution: {currentText}</Text>
        )}
      </SettingsRow>
      {camera.exposure_supported ? (
        <>
          <SettingsRow className={fields.row} data-toggle>
            <div className={layout.words}>
              <label htmlFor={exposureId} className={layout.label}>
                Automatic exposure
              </label>
              <div id={`${exposureId}-help`} className={layout.description}>
                Off uses the camera's own automatic exposure.
              </div>
            </div>
            <div className={classes.markRow}>
              <Switch
                id={exposureId}
                aria-label={`Automatic exposure for ${deviceName(camera.name)}`}
                aria-describedby={`${exposureId}-help`}
                checked={video.auto_exposure}
                disabled={disabled}
                onChange={event => save({ auto_exposure: event.currentTarget.checked })}
              />
              <SaveState status={statusFor('exposure')} />
            </div>
          </SettingsRow>
          <BrightnessTarget
            camera={camera}
            off={disabled}
            pending={pending}
            save={save}
            status={statusFor('brightness')}
          />
        </>
      ) : (
        camera.exposure_note && (
          <SettingsRow className={fields.row}>
            <span className={layout.label}>Automatic exposure</span>
            <Text size="sm" c="dimmed">
              {camera.exposure_note}
            </Text>
          </SettingsRow>
        )
      )}
    </>
  );
}

/** Which control in a card a save belongs to: the field it changed ("exposure"), and the camera or microphone it
 *  belongs to. Null is the card itself, for a change no single control owns: restarting all capture. */
type Mark = { key: string | null; status: SaveStatus };
const NO_MARK: Mark = { key: null, status: SAVE_IDLE };
/** The state to show in one control's slot: its own, or nothing. */
const markOf = (mark: Mark, key: string | null): SaveStatus => (mark.key === key ? mark.status : SAVE_IDLE);

/** The field of a camera change, as a mark's key: saving it is shown beside the control it was made on. */
function fieldOf(change: { index: number; on: boolean } | ({ index: number } & CameraVideoChange)): string {
  if ('on' in change) return `use:${change.index}`;
  if ('flip' in change) return `flip:${change.index}`;
  if ('auto_exposure' in change) return `exposure:${change.index}`;
  if ('exposure_target' in change) return `brightness:${change.index}`;
  return `resolution:${change.index}`;
}

function Cameras({
  view,
  off,
  mark,
  onStatus,
}: {
  view: DeviceView;
  off: boolean;
  mark: Mark;
  onStatus: (key: string | null, status: SaveStatus) => void;
}) {
  const d = useDraft();
  const states = useCameraStates();
  const prefs = usePrefs();
  const client = useQueryClient();
  const [zonesOf, setZonesOf] = useState<string | null>(null); // the camera whose zone editor is open
  const [restartFailed, setRestartFailed] = useState(false);
  const receive = (r: DeviceView, key: string | null) => {
    if (r.ok === false) return onStatus(key, failed(r.error || 'Not saved'));
    client.setQueryData(DEVICES_KEY, r);
    const restart = r.restart;
    if (restart && restart.status !== 'restarting') {
      onStatus(
        key,
        failed(
          restart.device === 'all'
            ? `Couldn't restart all capture. ${restart.message}`
            : `Couldn't restart ${restart.device}. Settings saved. ${restart.message}`,
        ),
      );
      setRestartFailed(!!r.restart_all_available);
    } else {
      onStatus(
        key,
        saved(
          restart
            ? restart.device === 'all'
              ? 'Restarting all capture.'
              : `Settings saved; restarting ${restart.device}.`
            : 'Settings saved.',
        ),
      );
      setRestartFailed(false);
    }
  };
  const save = useMutation({
    mutationFn: (change: { index: number; on: boolean } | ({ index: number } & CameraVideoChange)) =>
      'on' in change ? saveCamera(change.index, change.on) : saveCameraVideo(change),
    onMutate: change => {
      setRestartFailed(false);
      onStatus(
        fieldOf(change),
        saving('on' in change ? 'Saving; all cameras pause while capture restarts.' : 'Saving camera settings.'),
      );
    },
    onSuccess: (r, change) => receive(r, fieldOf(change)),
    onError: (e, change) => onStatus(fieldOf(change), failed(`Couldn't confirm the save: ${messageOf(e)}`)),
  });
  const restartAll = useMutation({
    mutationFn: restartCapture,
    onMutate: () => onStatus(null, saving('Restarting all capture; all cameras pause for a few seconds.')),
    onSuccess: r => receive(r, null),
    onError: e => onStatus(null, failed(`Couldn't restart all capture: ${messageOf(e)}`)),
  });
  const pending = save.isPending || restartAll.isPending;
  const places = parsePlaces(d.value('vision.camera_names'), shell().cams);
  // One camera at least: the last used one that is plugged in can't be left out (capture would have none)
  const watching = view.cameras.filter(c => c.present && c.selected);
  if (!view.cameras.length)
    return (
      <SettingsRow>
        <Text c="dimmed">{view.notes.cameras || 'No cameras found.'}</Text>
      </SettingsRow>
    );
  return (
    <>
      {restartFailed && (
        <SettingsRow>
          <Stack gap="xs" className={classes.restartFallback}>
            <Text size="xs" c="dimmed">
              Restarting all capture pauses every camera and microphone for a few seconds.
            </Text>
            <Button variant="default" onClick={() => restartAll.mutate()} disabled={off || pending}>
              Restart all capture
            </Button>
          </Stack>
        </SettingsRow>
      )}
      {view.cameras.map(c => {
        const cam = c.stream ?? `cam${c.index}`;
        const on = c.selected;
        const state = states && Object.hasOwn(states, cam) ? states[cam] : undefined;
        const status = cameraStatus({ present: c.present, selected: on }, state);
        return (
          <SourceItem
            key={c.index}
            on={on}
            aria-label={deviceName(c.name) || cam}
            head={
              <CameraHead
                cam={cam}
                label={deviceName(c.name) || cam}
                places={places}
                on={on}
                live={c.present && !!state}
                // off, the status still says when it's unplugged: why Use can't be checked
                status={on || !c.present ? status : null}
                use={{
                  description: 'Changing Use restarts all capture.',
                  onChange: checked => save.mutate({ index: c.index, on: checked }),
                  disabled:
                    off || pending || (!on && !c.present) || (watching.length === 1 && watching[0]?.index === c.index),
                  mark: <SaveState status={markOf(mark, `use:${c.index}`)} />,
                }}
              />
            }
          >
            <SettingsRow className={fields.row}>
              <span className={layout.label}>Picture and zones</span>
              <div className={classes.toggles}>
                <Checkbox
                  label="Mirror"
                  classNames={toggleClasses}
                  checked={mirrored(prefs, cam)}
                  onChange={e => setMirrored(cam, e.currentTarget.checked)}
                />
                <div className={classes.markRow} data-align="start">
                  <Checkbox
                    label="Upside down"
                    classNames={toggleClasses}
                    checked={!!c.flip}
                    onChange={e => save.mutate({ index: c.index, flip: e.currentTarget.checked })}
                    disabled={off || pending || !c.present}
                  />
                  <SaveState status={markOf(mark, `flip:${c.index}`)} />
                </div>
                <Button variant="default" size="xs" onClick={() => setZonesOf(cam)}>
                  Zones
                </Button>
              </div>
            </SettingsRow>
            <CameraName
              cam={cam}
              device={deviceName(c.name) || cam}
              saved={places[cam] ?? ''}
              off={off}
              onChange={name =>
                d.set('vision.camera_names', withPlace(d.value('vision.camera_names'), cam, name, shell().cams))
              }
            />
            <CameraSettings
              camera={c}
              off={off}
              pending={pending}
              save={change => save.mutate({ index: c.index, ...change })}
              statusFor={field => markOf(mark, `${field}:${c.index}`)}
            />
          </SourceItem>
        );
      })}
      {zonesOf && (
        <ZoneEditor
          cam={zonesOf}
          label={
            deviceName(view.cameras.find(c => (c.stream ?? `cam${c.index}`) === zonesOf)?.name ?? '') ||
            viewName(zonesOf)
          }
          onClose={() => setZonesOf(null)}
        />
      )}
    </>
  );
}

/** A sensitivity slider that saves when let go, at once and live: the microphone service follows pack.toml without
 *  a restart (changing mic sensitivity does not restart services). The reply is the new device list, so
 *  mics following the default move with it; Settings' copy of the config is refetched, so its draft agrees. */
function SensitivitySlider({
  label,
  thumb,
  value,
  save,
  off,
  reset,
  status,
  onStatus,
}: {
  label: string;
  thumb: string;
  value: number | null | undefined;
  save: (v: number) => Promise<DeviceView>;
  off: boolean;
  reset?: () => Promise<DeviceView>; // back to the default, for a mic with its own
  status: SaveStatus; // its own slot's state, at the end of its row
  onStatus: (status: SaveStatus) => void;
}) {
  const client = useQueryClient();
  const [shown, setShown] = useState(value ?? 70);
  const [was, setWas] = useState(value);
  if (value !== was) {
    setWas(value); // changed elsewhere (a new default): show it
    setShown(value ?? 70);
  }
  const mutation = useMutation({
    mutationFn: (v: number | null) => (v == null && reset ? reset() : save(v ?? 70)),
    onMutate: () => onStatus(saving('Saving the sensitivity.')),
    onSuccess: r => {
      if (r.ok === false) {
        onStatus(failed(r.error || 'Not saved'));
        return;
      }
      client.setQueryData(DEVICES_KEY, r);
      void client.invalidateQueries({ queryKey: CONFIG_KEY });
      onStatus(saved(r.note || 'Saved, in use now'));
    },
    onError: e => {
      onStatus(failed(`Not saved: ${messageOf(e)}`));
      setShown(value ?? 70);
    },
  });
  return (
    <div className={fields.row}>
      <span className={layout.label}>{label}</span>
      <div className={classes.sliderControl}>
        <Slider
          className={classes.slider}
          min={0}
          max={100}
          step={5}
          value={shown}
          onChange={setShown}
          onChangeEnd={v => mutation.mutate(v)}
          disabled={off || value == null}
          thumbLabel={thumb}
          label={v => `${v}`}
        />
        <span className={classes.value}>{shown}</span>
        {reset && (
          <Button variant="subtle" size="compact-xs" onClick={() => mutation.mutate(null)} disabled={off}>
            Use default
          </Button>
        )}
        <SaveState status={status} />
      </div>
    </div>
  );
}

/** Default microphone sensitivity in Devices: live, outside the draft. Its caller supplies the shared row, and this
 *  slider's own save state. */
export function DefaultSensitivity({
  row,
  id,
  disabled,
  status,
  onStatus,
}: ControlProps & { status: SaveStatus; onStatus: (status: SaveStatus) => void }) {
  const d = useDraft();
  const view = useDeviceView();
  const saved = view.data?.default_sensitivity ?? Number(d.value(row.key) ?? 70);
  return (
    <div id={id} tabIndex={-1} className={classes.defaultSensitivity}>
      <SensitivitySlider
        label="Default microphone sensitivity"
        thumb="Default sensitivity"
        value={saved}
        save={saveDefaultSensitivity}
        off={disabled || !d.view.control}
        status={status}
        onStatus={onStatus}
      />
    </div>
  );
}

/** Auto gain for one mic: on or off at once, live, outside the draft. Under it, which gain moves and where it
 *  is now, from the microphone service's status ("Hardware gain +12.5 dB": the mic's own capture volume). */
function AutoGain({
  mic,
  live,
  off,
  status,
  onStatus,
}: {
  mic: MicRow;
  live: MicGain | undefined;
  off: boolean;
  status: SaveStatus;
  onStatus: (status: SaveStatus) => void;
}) {
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: (on: boolean) => saveAutoGain(mic.id, on),
    onMutate: () => onStatus(saving('Saving the auto gain.')),
    onSuccess: r => {
      if (r.ok === false) return onStatus(failed(r.error || 'Not saved'));
      client.setQueryData(DEVICES_KEY, r);
      onStatus(saved('Auto gain saved.'));
    },
    onError: e => onStatus(failed(`Not saved: ${messageOf(e)}`)),
  });
  const now = gainText(live ?? mic.gain);
  return (
    <SettingsRow>
      <div className={classes.markRow} data-align="start">
        <Checkbox
          label="Auto gain"
          classNames={toggleClasses}
          description={mic.auto_gain ? now || 'Starts when the microphone service has it' : undefined}
          checked={!!mic.auto_gain}
          onChange={e => save.mutate(e.currentTarget.checked)}
          disabled={off || save.isPending}
        />
        <SaveState status={status} />
      </div>
      {/* On the Linux machine, a Wave:3 raised 40 dB matched its owner's lines to unnamed voices */}
      <Text size="sm" c="dimmed" mt={4}>
        Can change how voices are recognized. Record your voice sample again after turning it on.
      </Text>
    </SettingsRow>
  );
}

function Microphones({
  view,
  off,
  mark,
  onStatus,
}: {
  view: DeviceView;
  off: boolean;
  mark: Mark;
  onStatus: (key: string | null, status: SaveStatus) => void;
}) {
  const ambient = useLive('ambient');
  const root = useRef<HTMLDivElement>(null);
  const client = useQueryClient();
  const save = useMutation({
    mutationFn: ({ id, on }: { id: string; on: boolean }) => saveMic(id, on),
    onMutate: ({ id }) => onStatus(`listen:${id}`, saving('Saving, and restarting capture.')),
    onSuccess: (r, { id }) => {
      if (r.ok === false) return onStatus(`listen:${id}`, failed(r.error || 'Not saved'));
      client.setQueryData(DEVICES_KEY, r);
      onStatus(`listen:${id}`, saved('Microphone saved.'));
    },
    onError: (e, { id }) => onStatus(`listen:${id}`, failed(`Not saved: ${messageOf(e)}`)),
  });
  useLiveLevels(root); // the meters move with the live levels
  // The live status is keyed by each microphone's stream name (wave3), as capture's manifest names it
  const status = (ambient?.devices ?? {}) as Record<string, AmbientDevice>;
  // One microphone at least: capture would have none to listen with
  const only = view.mics.filter(m => m.present && m.selected).map(m => m.id);
  const known = new Map(view.mics.map(m => [m.id, m.name]));
  if (!view.mics.length)
    return (
      <SettingsRow>
        <Text c="dimmed">{view.notes.mics || 'No microphones found.'}</Text>
      </SettingsRow>
    );
  return (
    <div ref={root}>
      {view.mics.map(m => {
        const name = micName(m.id, known);
        const dev = m.stream ? status[m.stream] : undefined;
        const level = typeof dev?.level_dbfs === 'number' ? dev.level_dbfs : (m.level_dbfs ?? -120);
        const threshold = typeof dev?.threshold_dbfs === 'number' ? dev.threshold_dbfs : null;
        return (
          <SourceItem
            key={m.id}
            on={m.selected}
            aria-label={name}
            head={
              <SettingsRow className={`${fields.row} ${classes.mic}`}>
                <div className={classes.micName}>
                  <span className={layout.label}>{name}</span>
                  <Text span size="sm" c="dimmed" title={m.id}>
                    {m.present ? m.id : `${m.id} (not plugged in)`}
                  </Text>
                </div>
                <div className={classes.toggles}>
                  <div className={classes.markRow} data-align="start">
                    <Checkbox
                      label="Listen with this"
                      classNames={toggleClasses}
                      checked={m.selected}
                      onChange={e => save.mutate({ id: m.id, on: e.currentTarget.checked })}
                      disabled={off || save.isPending || !m.present || (only.length === 1 && only[0] === m.id)}
                    />
                    <SaveState status={markOf(mark, `listen:${m.id}`)} />
                  </div>
                </div>
              </SettingsRow>
            }
          >
            <SettingsRow>
              <div className={classes.micControls}>
                <meter
                  className={classes.srOnly}
                  min={0}
                  max={100}
                  value={Math.round(Number.parseFloat(levelWidth(level)))}
                  aria-label={`${name} level`}
                />
                <span
                  className={classes.meter}
                  aria-hidden="true"
                  data-level-key={m.present && dev ? (m.stream ?? undefined) : undefined}
                  title={threshold != null ? `Speech has to pass the mark (${threshold.toFixed(0)} dBFS)` : undefined}
                >
                  <b style={{ '--level': levelWidth(level) } as CSSProperties} />
                  {threshold != null && <s style={{ left: levelWidth(threshold) }} data-testid="threshold" />}
                  <i />
                </span>
                <SensitivitySlider
                  label="Sensitivity"
                  thumb={`Sensitivity for ${name}`}
                  value={m.sensitivity}
                  save={v => saveSensitivity(m.id, v)}
                  off={off}
                  reset={m.sensitivity_custom ? () => saveSensitivity(m.id, null) : undefined}
                  status={markOf(mark, `sensitivity:${m.id}`)}
                  onStatus={status => onStatus(`sensitivity:${m.id}`, status)}
                />
              </div>
            </SettingsRow>
            <AutoGain
              mic={m}
              live={dev?.gain}
              off={off || !m.present}
              status={markOf(mark, `autogain:${m.id}`)}
              onStatus={status => onStatus(`autogain:${m.id}`, status)}
            />
          </SourceItem>
        );
      })}
    </div>
  );
}

export function Devices() {
  const d = useDraft();
  const sensitivityRow = d.all.find(row => row.key === 'audio.sensitivity');
  const view = useDeviceView();
  const off = !d.view.control || d.busy;
  // A save is said beside the control it was made on, in that row's own slot. The cameras card keeps one in its
  // heading for the one change no single control owns: restarting all capture.
  const [camerasMark, setCamerasMark] = useState<Mark>(NO_MARK);
  const [micsMark, setMicsMark] = useState<Mark>(NO_MARK);
  return (
    <>
      <Text size="sm" c="dimmed">
        {/* Every card on the page: it named only two of the four */}
        On this page: <Anchor href={`#${DEVICE_CARD.cameras}`}>Cameras</Anchor> ·{' '}
        <Anchor href={`#${CARD_ID.camera}`}>Network cameras</Anchor> ·{' '}
        <Anchor href={`#${DEVICE_CARD.microphones}`}>Microphones</Anchor> ·{' '}
        <Anchor href={`#${DEVICE_CARD.bluetooth}`}>Bluetooth</Anchor>
      </Text>
      <SettingsSection
        title="Cameras"
        id={DEVICE_CARD.cameras}
        tabIndex={-1}
        trailing={<SaveState status={markOf(camerasMark, null)} />}
      >
        {view.data ? (
          <Cameras
            view={view.data}
            off={off}
            mark={camerasMark}
            onStatus={(key, status) => setCamerasMark({ key, status })}
          />
        ) : (
          <SettingsRow>
            <Text c="dimmed">
              {view.error ? `Could not list cameras: ${messageOf(view.error)}` : 'Looking for cameras…'}
            </Text>
          </SettingsRow>
        )}
      </SettingsSection>
      {/* Cameras recorded over the network: Wyze now, UniFi Protect and Reolink next (settings/sources/). */}
      <NetworkCameras
        names={{
          value: d.value('vision.camera_names'),
          set: value => d.set('vision.camera_names', value),
          disabled: off,
        }}
      />
      <SettingsSection title="Microphones" id={DEVICE_CARD.microphones} tabIndex={-1}>
        {sensitivityRow && (
          <SettingsRow>
            <DefaultSensitivity
              row={sensitivityRow}
              id="set-audio.sensitivity"
              labelled="default-sensitivity-label"
              disabled={off}
              status={markOf(micsMark, 'default-sensitivity')}
              onStatus={status => setMicsMark({ key: 'default-sensitivity', status })}
            />
          </SettingsRow>
        )}
        {d.all
          .filter(row => row.key === 'audio.auto_gain_target_dbfs')
          .map(row => (
            <SettingRow key={row.key} row={row} />
          ))}
        {view.data ? (
          <Microphones
            view={view.data}
            off={off}
            mark={micsMark}
            onStatus={(key, status) => setMicsMark({ key, status })}
          />
        ) : (
          <SettingsRow>
            <Text c="dimmed">
              {view.error ? `Could not list microphones: ${messageOf(view.error)}` : 'Looking for microphones…'}
            </Text>
          </SettingsRow>
        )}
      </SettingsSection>
      {/* Phones, watches, and tags linked to people and things live in Known › Devices now. */}
      <Text size="sm">
        Phones, watches, and tags are in{' '}
        <Anchor href={href('/devices')} onClick={linkClick}>
          Known › Devices
        </Anchor>
        , where you link them to people and things.
      </Text>
      {!d.view.control && (
        <Text size="sm" c="dimmed">
          Read only. Cameras and microphones can only be changed from the machine Liberational runs on.
        </Text>
      )}
    </>
  );
}
