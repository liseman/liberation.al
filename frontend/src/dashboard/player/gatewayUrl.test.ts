import { describe, expect, it } from 'vitest';
import { defaultGatewayUrl } from './protocol';

describe('defaultGatewayUrl', () => {
  it('uses the plain port for an http dashboard', () => {
    expect(defaultGatewayUrl({ hostname: 'mini.local', protocol: 'http:' })).toBe('ws://mini.local:8766/ws');
  });
  it('uses the TLS proxy port for an https dashboard (Tailscale)', () => {
    expect(defaultGatewayUrl({ hostname: 'mini.example.ts.net', protocol: 'https:' })).toBe(
      'wss://mini.example.ts.net:8767/ws',
    );
  });
  it('falls back to localhost with no hostname', () => {
    expect(defaultGatewayUrl({ hostname: '', protocol: '' })).toBe('ws://localhost:8766/ws');
  });
});
