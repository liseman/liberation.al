import { Badge, Button, Group, NativeSelect, Stack, Text, TextInput } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf, request } from '../../../shared/http';
import { type Message, Status } from '../../ingest/bits';
import { setActiveCameras, shell } from '../../shell';
import { DEVICES_KEY } from '../../sources/api';
import { CameraHead, CameraName } from '../devices/CameraCard';
import { useCameraStates } from '../devices/issues';
import { cameraStatus, parsePlaces, withPlace } from '../devices/model';
import { SettingsRow, SourceItem } from '../layout';
import { CONFIG_KEY, type ConfigView } from '../model';
import {
  addNetworkCamera,
  type CameraDiscovery,
  type CameraNamesDraft,
  type ConfiguredNetworkCamera,
  cameraName,
  type DiscoveredCamera,
  type DiscoverySource,
  discoverNetworkCameras,
  getNetworkCameras,
  NETWORK_CAMERAS_KEY,
  saveNetworkCameraUse,
  validCameraName,
} from './networkCameras';
import classes from './sources.module.css';

function Proposal({
  camera,
  disabled,
  added,
  captureReady,
  onAdd,
}: {
  camera: DiscoveredCamera;
  disabled: boolean;
  added: boolean;
  captureReady: boolean;
  onAdd: (stream: string, name: string) => Promise<void>;
}) {
  const label = camera.name || camera.model || camera.ip || 'Discovered camera';
  const [streamId, setStreamId] = useState(camera.streams.find(s => s.ready)?.id ?? camera.streams[0]?.id ?? '');
  const [name, setName] = useState(cameraName(camera.name || camera.model));
  const stream = camera.streams.find(s => s.id === streamId);
  return (
    <SourceItem
      component="li"
      aria-label={label}
      on
      head={
        <SettingsRow>
          <Group justify="space-between">
            <Text fw={500}>{label}</Text>
            <Badge variant="light">{camera.source}</Badge>
          </Group>
        </SettingsRow>
      }
    >
      <SettingsRow>
        <Stack gap="xs">
          {(camera.ip || camera.model) && (
            <Text size="sm" c="dimmed">
              {[camera.model, camera.ip].filter(Boolean).join(' · ')}
            </Text>
          )}
          {camera.requires_credentials && (
            <Text size="sm">Credentials are required. Save the camera credentials locally, then discover again.</Text>
          )}
          {camera.streams.length > 0 ? (
            <>
              <NativeSelect
                label={`Stream for ${label}`}
                value={streamId}
                onChange={e => setStreamId(e.currentTarget.value)}
                disabled={disabled || added}
                data={camera.streams.map(s => ({
                  value: s.id,
                  label: `${s.quality}${s.ready ? '' : ' (unavailable)'}`,
                }))}
              />
              {stream && (
                <Text size="xs" style={{ overflowWrap: 'anywhere' }}>
                  {stream.url_display}
                </Text>
              )}
              <TextInput
                label={`Camera name for ${label}`}
                value={name}
                onChange={e => setName(e.currentTarget.value)}
                maxLength={64}
                description="Start with a letter; use letters, numbers, hyphens, or underscores (up to 64 characters)."
                error={!validCameraName(name) ? 'Choose a valid camera name.' : undefined}
                disabled={disabled || added}
              />
            </>
          ) : (
            <Text size="sm" c="dimmed">
              No usable streams were returned.
            </Text>
          )}
          <Button
            size="xs"
            variant="default"
            disabled={disabled || added || !captureReady || !stream?.ready || !validCameraName(name)}
            onClick={() => void onAdd(streamId, name)}
          >
            {added ? 'Added' : `Add ${label}`}
          </Button>
        </Stack>
      </SettingsRow>
    </SourceItem>
  );
}

/** A network camera on the same card as a plugged-in one (devices/CameraCard.tsx): preview, Use, and Name, with
 *  where it streams from underneath. */
function ConfiguredCamera({
  camera,
  places,
  ids,
  names,
  disabled,
  notReady,
  onUse,
}: {
  camera: ConfiguredNetworkCamera;
  places: Readonly<Record<string, string>>;
  ids: string[];
  names?: CameraNamesDraft;
  disabled: boolean;
  notReady: boolean;
  onUse: (on: boolean) => void;
}) {
  const states = useCameraStates();
  const label = camera.name || camera.id;
  const state = states && Object.hasOwn(states, camera.id) ? states[camera.id] : undefined;
  return (
    <SourceItem
      component="li"
      aria-label={label}
      on={camera.enabled}
      head={
        <CameraHead
          cam={camera.id}
          label={label}
          places={places}
          on={camera.enabled}
          live={!!state}
          status={camera.enabled ? cameraStatus({ present: true, selected: true }, state) : null}
          use={{
            description: 'Changing Use restarts capture.',
            disabled: disabled || notReady || !!names?.disabled,
            onChange: onUse,
            note: notReady && (
              <Text size="xs" c="dimmed">
                Update capture before changing camera use.
              </Text>
            ),
          }}
        />
      }
    >
      {names && (
        <CameraName
          cam={camera.id}
          device={label}
          saved={places[camera.id] ?? ''}
          off={disabled || !!names.disabled}
          onChange={name => names.set(withPlace(names.value, camera.id, name, ids))}
        />
      )}
      <SettingsRow>
        <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
          {[camera.model, camera.ip, camera.url_display].filter(Boolean).join(' · ')}
        </Text>
      </SettingsRow>
    </SourceItem>
  );
}

export function NetworkCameraDiscovery({ names }: { names?: CameraNamesDraft } = {}) {
  const client = useQueryClient();
  // SettingsPage owns the permission query. This cache subscription never contacts an ingest worker.
  const owner = useQuery<ConfigView>({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
    enabled: false,
  });
  const control = owner.data?.control === true;
  const configured = useQuery({
    queryKey: NETWORK_CAMERAS_KEY,
    queryFn: ({ signal }) => getNetworkCameras(signal),
    retry: false,
  });
  const [source, setSource] = useState<DiscoverySource>('all');
  const [result, setResult] = useState<CameraDiscovery | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<Message>(null);
  const [added, setAdded] = useState<Set<string>>(new Set());
  const captureReady = result?.capture_ready === true && configured.data?.capture_ready !== false;
  const cameraIds = [...new Set([...shell().cams, ...(configured.data?.cameras.map(camera => camera.id) ?? [])])];
  const places = names ? parsePlaces(names.value, cameraIds) : shell().views;
  const changeUse = async (camera: ConfiguredNetworkCamera, enabled: boolean) => {
    if (!control || busy || names?.disabled || configured.data?.controls_ready !== true) return;
    setBusy(true);
    setMessage(null);
    try {
      const reply = await saveNetworkCameraUse(camera.id, enabled);
      setActiveCameras(reply.active_cameras);
      setMessage({
        text: reply.restarted.length
          ? 'Camera use saved. Capture is restarting; cameras and microphones pause briefly.'
          : 'Camera use is unchanged.',
      });
      await Promise.all([
        client.invalidateQueries({ queryKey: NETWORK_CAMERAS_KEY }),
        client.invalidateQueries({ queryKey: DEVICES_KEY }),
      ]);
    } catch (error) {
      setMessage({ err: `Could not change camera use: ${messageOf(error)}` });
      // The save can succeed before a restart request fails; refresh persisted state without inventing a rollback.
      await Promise.all([
        client.invalidateQueries({ queryKey: NETWORK_CAMERAS_KEY }),
        client.invalidateQueries({ queryKey: DEVICES_KEY }),
      ]);
    } finally {
      setBusy(false);
    }
  };
  const discover = async () => {
    if (!control || busy) return;
    setBusy(true);
    setResult(null);
    setAdded(new Set());
    setMessage(null);
    try {
      setResult(await discoverNetworkCameras(source));
    } catch (error) {
      setMessage({ err: `Could not discover cameras: ${messageOf(error)}` });
    } finally {
      setBusy(false);
    }
  };
  const add = async (camera: DiscoveredCamera, streamId: string, name: string) => {
    if (
      !result ||
      !control ||
      !captureReady ||
      !validCameraName(name) ||
      busy ||
      added.has(camera.id) ||
      !camera.streams.some(s => s.id === streamId && s.ready)
    )
      return;
    setBusy(true);
    setMessage(null);
    try {
      await addNetworkCamera(result.discovery_id, camera.id, streamId, name);
      setAdded(previous => new Set([...previous, camera.id]));
      setMessage({ text: 'Camera added.' });
      await Promise.all([
        client.invalidateQueries({ queryKey: NETWORK_CAMERAS_KEY }),
        client.invalidateQueries({ queryKey: DEVICES_KEY }),
      ]);
    } catch (error) {
      setMessage({ err: `Could not add camera: ${messageOf(error)}` });
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      {(configured.data?.cameras.length ?? 0) > 0 && (
        <ul className={classes.list} aria-label="Configured network cameras">
          {configured.data?.cameras.map(camera => (
            <ConfiguredCamera
              key={camera.id}
              camera={camera}
              places={places}
              ids={cameraIds}
              names={names}
              disabled={!control || busy}
              notReady={configured.data?.controls_ready !== true}
              onUse={checked => void changeUse(camera, checked)}
            />
          ))}
        </ul>
      )}
      <SettingsRow>
        <Stack gap="xs">
          {configured.isError && (
            <Text size="sm" c="red">
              Could not load network cameras: {messageOf(configured.error)}
            </Text>
          )}
          <Text size="sm" c="dimmed">
            Discover cameras through UniFi Protect and ONVIF. Discovery sends a local network probe only when you click
            Discover.
          </Text>
          <Group align="end">
            <NativeSelect
              label="Discovery source"
              value={source}
              onChange={e => setSource(e.currentTarget.value as DiscoverySource)}
              disabled={!control || busy}
              data={[
                { value: 'all', label: 'UniFi Protect and ONVIF' },
                { value: 'protect', label: 'UniFi Protect' },
                { value: 'onvif', label: 'ONVIF' },
              ]}
            />
            <Button
              size="xs"
              variant="default"
              disabled={!control || busy}
              loading={busy}
              onClick={() => void discover()}
            >
              Discover
            </Button>
          </Group>
          {!control && (
            <Text size="sm" c="dimmed">
              Camera discovery and changes require control from this computer.
            </Text>
          )}
          {(configured.data?.capture_ready === false || result?.capture_ready === false) && (
            <Text size="sm" c="dimmed">
              Network camera capture is not available on this machine. You can discover cameras, but cannot add them
              yet.
            </Text>
          )}
          <Status msg={message} />
          {[...new Set(result?.warnings ?? [])].map(warning => (
            <Text key={warning} size="sm" c="dimmed">
              {warning}
            </Text>
          ))}
          {result && result.cameras.length === 0 && (
            <Text size="sm">No cameras responded. Cameras that did not respond remain unknown.</Text>
          )}
        </Stack>
      </SettingsRow>
      {result && result.cameras.length > 0 && (
        <ul className={classes.list} aria-label="Discovered network cameras">
          {result.cameras.map(camera => (
            <Proposal
              key={`${result.discovery_id}-${camera.id}`}
              camera={camera}
              added={added.has(camera.id)}
              disabled={!control || busy}
              captureReady={captureReady}
              onAdd={(stream, name) => add(camera, stream, name)}
            />
          ))}
        </ul>
      )}
    </>
  );
}
