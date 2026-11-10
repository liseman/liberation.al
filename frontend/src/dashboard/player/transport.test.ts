import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  connect,
  connectWebSocket,
  connectWebTransport,
  forgetWtInfoForTests,
  type Handlers,
  hexToBytes,
  httpBase,
  WT_INFO_MS,
} from './transport';

beforeEach(() => forgetWtInfoForTests());

class FakeSocket {
  static all: FakeSocket[] = [];
  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readyState = 0;
  binaryType = 'blob';
  sent: string[] = [];
  onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  private opens: (() => void)[] = [];
  readonly url: string;
  constructor(url: string) {
    this.url = url;
    FakeSocket.all.push(this);
  }
  addEventListener(type: string, fn: () => void) {
    if (type === 'open') this.opens.push(fn);
  }
  send(s: string) {
    this.sent.push(s);
  }
  close() {
    this.onclose?.({ code: 1000, reason: 'bye' } as CloseEvent);
  }
  open() {
    this.readyState = 1;
    for (const fn of this.opens.splice(0)) fn();
  }
}

function handlers(): Handlers & { replies: unknown[]; binaries: ArrayBuffer[]; closed: string[] } {
  const h = {
    replies: [] as unknown[],
    binaries: [] as ArrayBuffer[],
    closed: [] as string[],
    onReply: (r: unknown) => h.replies.push(r),
    onBinary: (b: ArrayBuffer) => h.binaries.push(b),
    onClose: (r: string) => h.closed.push(r),
  };
  return h;
}

describe('the WebSocket transport', () => {
  it('queues a request until the socket opens, routes text and binary, and reports close', () => {
    const h = handlers();
    const env = { WebSocket: FakeSocket as unknown as typeof WebSocket, fetch: vi.fn() as unknown as typeof fetch };
    const c = connectWebSocket('ws://h:8766/ws', h, env);
    const s = FakeSocket.all.at(-1);
    if (!s) throw new Error('no socket');
    expect(s.binaryType).toBe('arraybuffer');
    c.send({ type: 'streams' });
    expect(s.sent).toEqual([]);
    s.open();
    expect(s.sent).toEqual(['{"type":"streams"}']);
    c.send({ type: 'ping' });
    expect(s.sent).toHaveLength(2);
    s.onmessage?.({ data: '{"type":"pong"}' } as MessageEvent);
    const buf = new ArrayBuffer(16);
    s.onmessage?.({ data: buf } as MessageEvent);
    expect(h.replies).toEqual([{ type: 'pong' }]);
    expect(h.binaries).toEqual([buf]);
    c.close();
    expect(h.closed).toEqual(['bye']);
    expect(c.transport).toBe('websocket');
  });

  it('falls back to WebSockets when the gateway has WebTransport off, and respects prefer', async () => {
    const h = handlers();
    const fetch = vi.fn(async () => ({ ok: true, json: async () => null }) as Response); // off
    const env = {
      WebSocket: FakeSocket as unknown as typeof WebSocket,
      WebTransport: class {} as never,
      fetch: fetch as unknown as typeof fetch,
    };
    const c = await connect('ws://h:8766/ws', h, 'auto', env);
    expect(fetch).toHaveBeenCalledWith('http://h:8766/wt');
    expect(c.transport).toBe('websocket');
    await expect(connect('ws://h:8766/ws', h, 'webtransport', env)).rejects.toThrow(/off/);
    const noWt = { WebSocket: FakeSocket as unknown as typeof WebSocket, fetch: fetch as unknown as typeof fetch };
    expect((await connect('wss://h:8766/ws', h, 'websocket', noWt)).transport).toBe('websocket');
    await expect(connect('ws://h:8766/ws', h, 'webtransport', noWt)).rejects.toThrow(/no WebTransport/);
  });

  it('asks a gateway whether WebTransport is on once a minute at most, and again after a failed ask', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    try {
      const h = handlers();
      let fail = true;
      const fetch = vi.fn(async () => {
        if (fail) throw new Error('down');
        return { ok: true, json: async () => null } as Response;
      });
      const env = {
        WebSocket: FakeSocket as unknown as typeof WebSocket,
        WebTransport: class {} as never,
        fetch: fetch as unknown as typeof fetch,
      };
      await connect('ws://h:8766/ws', h, 'auto', env); // the ask fails: WebSockets, and nothing remembered
      fail = false;
      for (let i = 0; i < 5; i++) await connect('ws://h:8766/ws', h, 'auto', env);
      expect(fetch).toHaveBeenCalledTimes(2);
      await connect('ws://other:8766/ws', h, 'auto', env); // another gateway asks for itself
      expect(fetch).toHaveBeenCalledTimes(3);
      vi.advanceTimersByTime(WT_INFO_MS);
      await connect('ws://h:8766/ws', h, 'auto', env);
      expect(fetch).toHaveBeenCalledTimes(4);
    } finally {
      vi.useRealTimers();
    }
  });

  it('hands WebTransport messages over in the order their streams were opened, not finished', async () => {
    const h = handlers();
    const controllers: ReadableStreamDefaultController<Uint8Array>[] = [];
    const message = () => new ReadableStream<Uint8Array>({ start: c => void controllers.push(c) });
    const first = message();
    const second = message();
    const incoming = new ReadableStream<ReadableStream<Uint8Array>>({
      start(c) {
        c.enqueue(first);
        c.enqueue(second);
        c.close();
      },
    });
    class FakeTransport {
      ready = Promise.resolve();
      closed = new Promise(() => {});
      incomingUnidirectionalStreams = incoming;
      async createBidirectionalStream() {
        return { readable: new ReadableStream<Uint8Array>(), writable: new WritableStream<Uint8Array>() };
      }
      close() {}
    }
    const env = { WebSocket: FakeSocket as unknown as typeof WebSocket, WebTransport: FakeTransport, fetch };
    const c = await connectWebTransport({ url: 'https://h:8766/wt', cert_hash: '' }, h, env);
    expect(c.transport).toBe('webtransport');
    const settle = () => new Promise(r => setTimeout(r, 0));
    await settle();
    // The second message finishes first.
    controllers[1]?.enqueue(new Uint8Array([2]));
    controllers[1]?.close();
    await settle();
    expect(h.binaries).toEqual([]);
    controllers[0]?.enqueue(new Uint8Array([1]));
    controllers[0]?.close();
    await settle();
    await settle();
    expect(h.binaries.map(b => Array.from(new Uint8Array(b)))).toEqual([[1], [2]]);
  });

  it('derives the http base and parses certificate hashes', () => {
    expect(httpBase('ws://devbox:8766/ws')).toBe('http://devbox:8766');
    expect(httpBase('wss://devbox:8766/ws?x=1')).toBe('https://devbox:8766');
    expect(Array.from(hexToBytes('00ff10'))).toEqual([0, 255, 16]);
  });
});
