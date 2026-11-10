// What the gateway's `streams` reply lists (continuum-gateway's source.rs StreamSummary).
export interface StreamSummary {
  id: number;
  /** The registered name, when the broker is in the loop (empty from the chunk directory alone). */
  name?: string;
  /** A producer is connected for it now (false from the chunk directory alone). */
  live?: boolean;
  format: string;
  /** The earliest and latest recorded time, microseconds UTC; both 0 for a stream that holds nothing. */
  start_us: number;
  end_us: number;
}

/** The gateway's plain WebSocket port when the page isn't told one, and the port a TLS proxy
 *  (`tailscale serve --https=8767`, tailnet only) puts in front of it for a dashboard opened over HTTPS: a page on
 *  https can't open ws://, and the gateway speaks plain WebSockets. */
export const GATEWAY_PORT = 8766;
export const GATEWAY_TLS_PORT = 8767;

/** The gateway's port as the dashboard filled it into the page (ports.gateway; the demo points it nowhere), read
 *  from window.NAMES directly so a module a worker imports doesn't pull in the shell. */
function gatewayPort(): number {
  const given = (globalThis as { NAMES?: { gatewayPort?: unknown } }).NAMES?.gatewayPort;
  return typeof given === 'number' ? given : GATEWAY_PORT;
}

/** The gateway's address next to the dashboard: the same host, on ports.gateway (or the TLS proxy's port). */
export function defaultGatewayUrl(location: { hostname: string; protocol: string }): string {
  const tls = location.protocol === 'https:';
  return `${tls ? 'wss' : 'ws'}://${location.hostname || 'localhost'}:${tls ? GATEWAY_TLS_PORT : gatewayPort()}/ws`;
}

/** One immutable chunk snapshot, identified independently of capture time. */
export interface MediaContext {
  type: 'media';
  context_id: number;
  stream: number;
  chunk_id: string | null;
  revision: number | null;
  profile_status: 'absent' | 'unsupported' | 'unavailable';
  native_geometry: { width: number; height: number } | null;
}
