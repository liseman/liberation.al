import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { newQueryClient, Providers } from '../../shared/Providers';
import { FakeEventSource } from '../camera/testFakes';
import { liveKey } from '../live/feed';
import { DEVICES_KEY } from '../sources/api';
import { Mics, micName } from './Mics';

const WAVE = 'alsa_input.usb-Elgato_Systems_Elgato_Wave_3_A017A41910AAFN-00.mono-fallback';

describe('micName', () => {
  it.each([
    [WAVE, 'Elgato Wave 3'],
    ['alsa_input.usb-Acme_Mic-00.mono-fallback', 'Acme Mic'],
    ['alsa_input.pci-0000_00_1f.3.analog-stereo', 'Built-in microphone'],
    ['alsa_input.usb-BEHRINGER_UMC204HD_192k-00.analog-stereo', 'BEHRINGER UMC204HD 192k'],
    ['alsa_input.usb-Audio-Technica_AT2020USB_-00.analog-stereo', 'Audio-Technica AT2020USB'],
    ['MacBook Pro Microphone', 'MacBook Pro Microphone'],
  ])('%s is %s', (raw, name) => {
    expect(micName(raw)).toBe(name);
  });

  it("uses the Sources picker's name when it has one", () => {
    expect(micName(WAVE, new Map([[WAVE, 'Room mic']]))).toBe('Room mic');
  });
});

describe('Mics (#353)', () => {
  beforeEach(() => {
    FakeEventSource.all = [];
    vi.stubGlobal('EventSource', FakeEventSource);
  });
  afterEach(() => vi.unstubAllGlobals());

  function renderMics(devices: Record<string, unknown>, picker?: unknown, compact = false) {
    const client = newQueryClient();
    client.setQueryData(liveKey('ambient'), { ts: 1, devices });
    if (picker) client.setQueryData(DEVICES_KEY, picker);
    return render(
      <Providers client={client} test>
        <Mics compact={compact} />
      </Providers>,
    );
  }

  it('shows a chip per microphone with its level and the mark speech has to pass', () => {
    renderMics({ wave: { name: WAVE, level_dbfs: -45, threshold_dbfs: -75, healthy: true } });
    const meter = screen.getByRole('meter', { name: 'Elgato Wave 3 level' });
    expect(meter).toHaveAttribute('value', '50'); // -45 dBFS on a -90 to 0 meter
    expect(screen.getByTestId('mic-threshold').style.left).toBe('16.7%');
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
  });

  it("shows auto gain's gain now, and nothing while auto gain is off (#410)", () => {
    const gain = { auto: true, kind: 'hardware', db: 12.5 };
    const { unmount } = renderMics({ wave: { name: WAVE, level_dbfs: -45, healthy: true, gain } });
    expect(screen.getByText('+12.5 dB')).toHaveAttribute('title', 'Hardware gain +12.5 dB');
    unmount();
    renderMics({ wave: { name: WAVE, level_dbfs: -45, healthy: true, gain: { ...gain, auto: false } } });
    expect(screen.queryByText('+12.5 dB')).not.toBeInTheDocument();
  });

  it("says the level is unavailable in the conversation's compact line until the levels arrive (#2486)", () => {
    renderMics({ wave: { name: WAVE, level_dbfs: -60, healthy: true } }, undefined, true);
    expect(screen.getByText('Listening · level unavailable')).toBeInTheDocument();
    const [stream] = FakeEventSource.all;
    stream?.send({ levels: { wave: { rms: -80, peak: -70, speech: false } } });
    expect(screen.getByText('Listening · quiet')).toBeInTheDocument();
    stream?.send({ levels: { wave: { rms: -50, peak: -40, speech: true } } });
    expect(screen.getByText('Listening · sound')).toBeInTheDocument();
  });

  it('shows sound detected while the energy gate is active, and quiet again after', () => {
    renderMics({ wave: { name: WAVE, level_dbfs: -60, healthy: true } });
    const [stream] = FakeEventSource.all;
    expect(stream?.url).toBe('/audio/levels');
    stream?.send({ levels: { wave: { rms: -50, peak: -40, speech: true } } });
    expect(screen.getByText('Sound detected')).toBeInTheDocument();
    stream?.send({ levels: { wave: { rms: -80, peak: -70, speech: false } } });
    expect(screen.getByText('Quiet')).toBeInTheDocument();
  });

  it('shows unavailable when live levels have no activity signal or the stream closes', () => {
    renderMics({ wave: { name: WAVE, healthy: true } });
    const [stream] = FakeEventSource.all;
    stream?.send({ levels: { wave: { rms: -50, peak: -40, speech: true } } });
    expect(screen.getByText('Sound detected')).toBeInTheDocument();
    stream?.send({ levels: { wave: { rms: -50, peak: -40 } } });
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
    stream?.send({ levels: { wave: { rms: -50, peak: -40, speech: false } } });
    expect(screen.getByText('Quiet')).toBeInTheDocument();
    stream?.send({ levels: {} });
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
  });

  it('expires an activity state when the levels connection stops delivering', async () => {
    vi.useFakeTimers();
    try {
      renderMics({ wave: { name: WAVE, healthy: true } });
      FakeEventSource.all[0]?.send({ levels: { wave: { rms: -50, peak: -40, speech: false } } });
      expect(screen.getByText('Quiet')).toBeInTheDocument();
      await act(() => vi.advanceTimersByTimeAsync(3000));
      expect(screen.getByText('Unavailable')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('shares one levels stream between the chips', () => {
    renderMics({ a: { name: 'Mic A', healthy: true }, b: { name: 'Mic B', healthy: true } });
    expect(FakeEventSource.all.filter(s => !s.closed)).toHaveLength(1);
  });

  it("says a microphone the service can't read isn't responding", () => {
    renderMics({ wave: { name: WAVE, healthy: false } });
    expect(screen.getByText('Not responding')).toBeInTheDocument();
  });

  it("uses the name the Sources picker found when it's known", () => {
    renderMics({ wave: { name: WAVE, healthy: true } }, { control: true, mics: [{ id: WAVE, name: 'Room mic' }] });
    expect(screen.getByRole('meter', { name: 'Room mic level' })).toBeInTheDocument();
  });

  it('shows nothing without microphones', () => {
    const { container } = renderMics({});
    expect(container.querySelector('.mics')).toBeNull();
  });
});
