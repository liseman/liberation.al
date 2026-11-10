// Native cameras use their own manifest API. URLs stay server-private; proposals use opaque IDs only.
import { request } from '../../../shared/http';

export const NETWORK_CAMERAS_KEY = ['network-cameras'] as const;
export type DiscoverySource = 'all' | 'protect' | 'onvif';
export interface ConfiguredNetworkCamera {
  id: string;
  name: string;
  source: 'rtsp';
  url_display: string;
  enabled: boolean;
  model?: string;
  ip?: string;
}
export interface NetworkCameraView {
  cameras: ConfiguredNetworkCamera[];
  capture_ready: boolean;
  controls_ready?: boolean;
}
export interface CameraNamesDraft {
  value: string | boolean | undefined;
  set: (value: string) => void;
  disabled?: boolean;
}
export interface DiscoveredStream {
  id: string;
  quality: string;
  url_display: string;
  ready: boolean;
}
export interface DiscoveredCamera {
  id: string;
  source: string;
  name?: string | null;
  model?: string | null;
  ip?: string | null;
  streams: DiscoveredStream[];
  requires_credentials: boolean;
}
export interface CameraDiscovery {
  discovery_id: string;
  cameras: DiscoveredCamera[];
  warnings: string[];
  capture_ready: boolean;
}
export const getNetworkCameras = (signal?: AbortSignal) =>
  request<NetworkCameraView>('/network-cameras', undefined, signal);
export const saveNetworkCameraUse = (id: string, enabled: boolean) =>
  request<{ ok: true; restarted: string[]; active_cameras: string[] }>('/network-cameras/use', { id, enabled });
export const discoverNetworkCameras = (source: DiscoverySource) =>
  request<CameraDiscovery>('/network-cameras/discover', { source });
export const addNetworkCamera = (discovery_id: string, camera_id: string, stream_id: string, name?: string) =>
  request<{ ok: true }>('/network-cameras/add', {
    discovery_id,
    camera_id,
    stream_id,
    ...(name?.trim() ? { name: name.trim() } : {}),
  });

/** Manifest identifiers are distinct from friendly discovery labels. */
export function cameraName(label: string | null | undefined): string {
  const slug =
    (label ?? '')
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^[-_]+|[-_]+$/g, '') || 'network-camera';
  return (/^[a-z]/.test(slug) ? slug : `camera-${slug}`).slice(0, 64);
}
export const validCameraName = (name: string) => /^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(name.trim());
