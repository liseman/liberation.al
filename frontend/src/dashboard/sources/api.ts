// The dashboard's source picker (dashboard/devices.py) and machine load (GET /stats).
import { request } from '../../shared/http';

/** A microphone, by its device name (id). */
export interface MicRow {
  id: string;
  name: string;
  host_api?: string | null;
  present: boolean;
  selected: boolean; // a microphone in capture's manifest (config/devices.toml) matches it
  stream?: string | null; // that microphone's name: its <stream>.audio on the broker, and the live status's key
  in_use: boolean;
  level_dbfs?: number | null;
  /** 0 to 100 (audio/sensitivity.py), or null when its setting is unreadable. */
  sensitivity?: number | null;
  /** Whether this mic has its own sensitivity, rather than audio.sensitivity. */
  sensitivity_custom?: boolean;
  /** Auto gain is on for this mic ([audio.mic_auto_gain]). */
  auto_gain?: boolean;
  /** Its gain as the microphone service last said, or null while it isn't running the mic. */
  gain?: MicGain | null;
}

/** A mic's gain (audio/gain.py): its own capture volume ("hardware") or a gain on the samples ("software"). */
export interface MicGain {
  auto: boolean;
  kind: 'hardware' | 'software' | null;
  db: number | null;
}

/** "Hardware gain +12.5 dB": which gain auto gain moves, and where it is now. */
export function gainText(g: MicGain | null | undefined): string {
  if (!g?.kind) return '';
  const kind = g.kind === 'hardware' ? 'Hardware gain' : 'Software gain';
  return g.db == null ? kind : `${kind} ${g.db > 0 ? '+' : ''}${g.db.toFixed(1)} dB`;
}

/** A camera, by its index (what vision.cams lists). */
export interface CameraRow {
  flip?: boolean;
  index: number;
  name: string;
  path?: string | null;
  usb?: string | null;
  present: boolean;
  selected: boolean; // a camera in capture's manifest (config/devices.toml) matches it
  stream?: string | null; // that camera's name: its <stream>.video on the broker
  in_use: boolean;
  video?: CameraVideo;
  modes?: CameraModes;
  exposure_supported?: boolean;
  exposure_note?: string;
}

export interface CameraMode {
  width: number;
  height: number;
  fps: number;
}

export interface CameraVideo extends CameraMode {
  format: string;
  auto_exposure: boolean;
  exposure_target: number;
}

export interface CameraModes {
  status: string;
  choices: { width: number; height: number; fps: number[] }[];
  diagnostics: string[];
}

export type CameraVideoChange =
  | CameraMode
  | { auto_exposure: boolean }
  | { exposure_target: number }
  | { flip: boolean };

export interface CaptureRestart {
  status: 'restarting' | 'unavailable' | 'failed';
  device: string;
  message: string;
}

export interface Devices {
  ok?: boolean;
  error?: string;
  control: boolean; // false: viewed from another machine, so read only
  mics: MicRow[];
  cameras: CameraRow[];
  notes: { mics?: string; cameras?: string };
  default_sensitivity?: number | null; // audio.sensitivity now, for every mic without its own
  restarted?: string[]; // after a save: the services restarted to pick it up
  note?: string; // after a save: what went wrong, or what overrides it
  restart?: CaptureRestart;
  restart_all_available?: boolean;
}

export interface Proc {
  pid?: number;
  role: string; // the service's name, or ollama, or ollama runner
  cpu: number; // percent of one core: 200 is two cores busy
  rss: number;
  model?: string | null; // an Ollama runner's model, when /api/ps names it
}

/** A model Ollama has loaded (GET /api/ps). */
export interface LoadedModel {
  name: string;
  size: number;
  size_vram?: number | null; // how much of it is in GPU memory
  ctx?: number | null; // its context length, in tokens
  params?: string | null; // "4.0B"
  purpose?: string | null;
  device?: string | null;
  since?: number | null; // only when reported by the source, never inferred from expiry or process uptime
}

/** A model reported by the process that loaded it. Times are Unix seconds; null means unreported. */
export interface RuntimeModel {
  key: string;
  name: string;
  purpose: string;
  device: string;
  since: number | null;
}

/** Memory belongs to the process as a whole, including any non-model allocations. */
export interface ModelProcess {
  pid: number;
  service: string;
  since: number | null;
  rss_bytes: number | null;
  gpu_bytes: number | null;
  models: RuntimeModel[];
}

export interface ConfiguredModel {
  key: string;
  name: string;
  purpose: string;
  device: string;
  state: 'unloaded' | 'unknown';
}

export interface ModelRuntime {
  processes: ModelProcess[];
  configured: ConfiguredModel[];
  errors: string[];
}

export interface Stats {
  error?: string;
  cpu?: number;
  cpu_count?: number;
  load?: number[];
  mem_used?: number;
  mem_total?: number;
  swap_used?: number;
  ollama?: {
    models?: LoadedModel[];
    error?: string | null;
    /** Whether Ollama answers, the configured models it lacks, and what to tell the owner. */
    up?: boolean;
    missing?: string[];
    message?: string | null;
  };
  procs?: Proc[];
  /** Application loader reports grouped by owning process. Ollama remains its own source above. */
  model_runtime?: ModelRuntime;
  /** System packages this machine is missing (config/system-packages.toml) and the commands that install them. */
  system?: {
    missing?: { name: string; why: string; optional: boolean; services: string[]; command?: string }[];
    commands?: string[];
    error?: string;
  };
}

export const DEVICES_KEY = ['devices'] as const;
export const STATS_KEY = ['stats'] as const;

export const getDevices = ({ signal }: { signal?: AbortSignal } = {}) =>
  request<Devices>('/devices', undefined, signal);

/** One mic's sensitivity, 0 to 100, or null for audio.sensitivity. Ambient applies it live. */
export const saveSensitivity = (mic: string, sensitivity: number | null) =>
  request<Devices>('/devices/sensitivity', { mic, sensitivity });

/** Auto gain on or off for one mic: live, nothing restarts. */
export const saveAutoGain = (mic: string, on: boolean) => request<Devices>('/devices/auto_gain', { mic, on });

/** Listen with one microphone or stop: edits capture's manifest and restarts capture. */
export const saveMic = (id: string, on: boolean) => request<Devices>('/devices/mic', { id, on });

/** Use one camera or stop using it: edits capture's manifest and restarts capture. */
export const saveCamera = (index: number, on: boolean) => request<Devices>('/devices/camera', { index, on });

/** audio.sensitivity, for every mic without its own: live, nothing restarts. */
export const saveDefaultSensitivity = (value: number) => request<Devices>('/devices/sensitivity', { default: value });

export const getStats = ({ signal }: { signal?: AbortSignal } = {}) => request<Stats>('/stats', undefined, signal);

/** One sample of what each service used: processor as % of the machine, memory and GPU memory in bytes, each
 *  with "other" as the rest of the machine. */
export interface UsageSample {
  ts: number;
  cpu: Record<string, number>;
  mem: Record<string, number>;
  gpu: { index: number; used: number; util: number | null; apps: Record<string, number> }[];
}

/** GET /stats/history: the samples newer than `since`, and what each resource stacks up to. */
export interface UsageHistory {
  step: number;
  window: number;
  cpu_count: number;
  mem_total: number;
  gpus: { index: number; name: string; total: number }[];
  samples: UsageSample[];
}

export const getUsageHistory = (since: number, { signal }: { signal?: AbortSignal } = {}) =>
  request<UsageHistory>(`/stats/history?since=${since}`, undefined, signal);

/** Rotate one capture camera, independently of browser mirroring. */
export const saveCameraFlip = (index: number, flip: boolean) => request<Devices>('/devices/camera', { index, flip });

export const saveCameraVideo = (change: { index: number } & CameraVideoChange) =>
  request<Devices>('/devices/camera', change);

/** Explicit fallback after a selective restart fails; all capture devices briefly pause. */
export const restartCapture = () => request<Devices>('/devices/capture/restart', {});
