import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse, http } from 'msw';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderPage } from '../../test/render';
import { server } from '../../test/server';
import { ServiceChecks } from './ServiceChecks';

const services = [
  {
    name: 'voice',
    description: 'Listens and speaks',
    default: true,
    state: 'off',
    installed: false,
    pids: [],
    selected: true,
  },
  {
    name: 'phone',
    description: 'Answers calls',
    default: false,
    state: 'off',
    installed: false,
    pids: [],
    selected: false,
  },
  {
    name: 'whisper',
    description: 'Transcribes audio',
    default: false,
    state: 'off',
    installed: false,
    pids: [],
    selected: false,
  },
];

function serve() {
  server.use(http.get('/services', () => HttpResponse.json({ ok: true, control: true, services })));
}

function Form({
  onChange,
  readOnly = false,
  initial = {},
}: {
  onChange: (patch: Record<string, string | boolean>) => void;
  readOnly?: boolean;
  initial?: Record<string, string | boolean>;
}) {
  const [values, setValues] = useState<Record<string, string | boolean>>({
    'services.enable': 'foreign-service',
    'services.disable': 'foreign-default',
    ...initial,
  });
  return (
    <ServiceChecks
      values={values}
      readOnly={readOnly}
      onChange={patch => {
        setValues(old => ({ ...old, ...patch }));
        onChange(patch);
      }}
    />
  );
}

describe('Settings service selection', () => {
  it('edits default and optional overrides without dropping another platform services', async () => {
    serve();
    const onChange = vi.fn();
    renderPage(<Form onChange={onChange} />);
    const voice = await screen.findByRole('checkbox', { name: 'voice' });
    const phone = screen.getByRole('checkbox', { name: 'phone' });
    expect(voice).toBeChecked();
    expect(phone).not.toBeChecked();
    await userEvent.click(voice);
    expect(onChange).toHaveBeenLastCalledWith({
      'services.enable': 'foreign-service',
      'services.disable': 'foreign-default, voice',
    });
    await userEvent.click(phone);
    expect(onChange).toHaveBeenLastCalledWith({
      'services.enable': 'foreign-service, phone',
      'services.disable': 'foreign-default, voice',
    });
    await userEvent.click(voice);
    expect(onChange).toHaveBeenLastCalledWith({
      'services.enable': 'foreign-service, phone',
      'services.disable': 'foreign-default',
    });
    await userEvent.click(phone);
    expect(onChange).toHaveBeenLastCalledWith({
      'services.enable': 'foreign-service',
      'services.disable': 'foreign-default',
    });
  });

  it('shows Whisper required by the draft ASR engine unless explicitly disabled', async () => {
    serve();
    const view = renderPage(<Form onChange={vi.fn()} initial={{ 'asr.engine': 'whisper' }} />);
    expect(await screen.findByRole('checkbox', { name: 'whisper' })).toBeChecked();
    expect(screen.getByText(/on because ASR uses Whisper/)).toBeInTheDocument();
    view.unmount();
    renderPage(<Form onChange={vi.fn()} initial={{ 'asr.engine': 'whisper', 'services.disable': 'whisper' }} />);
    expect(await screen.findByRole('checkbox', { name: 'whisper' })).not.toBeChecked();
  });

  it('disables all selection controls for a read-only viewer', async () => {
    serve();
    const onChange = vi.fn();
    renderPage(<Form onChange={onChange} readOnly />);
    const phone = await screen.findByRole('checkbox', { name: 'phone' });
    expect(phone).toBeDisabled();
    await userEvent.click(phone);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('shows load failures and lets the viewer retry', async () => {
    server.use(
      http.get('/services', () => HttpResponse.json({ error: 'Service manager unavailable' }, { status: 503 })),
    );
    renderPage(<Form onChange={vi.fn()} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Service manager unavailable');
    serve();
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.getByRole('checkbox', { name: 'voice' })).toBeChecked());
  });
});
