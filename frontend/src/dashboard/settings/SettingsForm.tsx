// Schema-driven Settings and first-run setup. Server values are query data; unsaved drafts belong to this form.
import {
  Accordion,
  Alert,
  Anchor,
  Badge,
  Button,
  Checkbox,
  Code,
  Fieldset,
  Group,
  PasswordInput,
  Radio,
  Stack,
  TagsInput,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { modals } from '@mantine/modals';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type ReactNode, useEffect, useId, useRef, useState } from 'react';
import { HttpError, messageOf, request } from '../../shared/http';
import { confirm } from '../lib/confirm';
import { PageError } from '../lib/page';
import { SettingsRow } from './layout';
import layout from './layout.module.css';
import {
  blank,
  CONFIG_KEY,
  type ConfigReply,
  type ConfigView,
  initialValues,
  isRequired,
  nice,
  restartDestination,
  type Setting,
  type Values,
} from './model';
import { followRestart } from './restart';
import { ServiceChecks } from './ServiceChecks';
import { FEATURES, itemOf, loadOpen, type SectionState, saveOpen, sectionState } from './sections';
import classes from './settings.module.css';
import { VoiceRef } from './VoiceRef';

export interface SettingsFormProps {
  setup?: boolean;
  onClose?: () => void;
  closeRequest?: number;
  onDirtyChange?: (dirty: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
}

export function SettingsForm(props: SettingsFormProps) {
  const config = useQuery({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
  });
  if (!config.data)
    return (
      <Stack>
        <Text role={config.error ? 'alert' : 'status'}>
          {config.error ? `Could not load the settings: ${messageOf(config.error)}` : 'Loading settings'}
        </Text>
        {config.error && (
          <Button variant="default" onClick={() => void config.refetch()}>
            Try again
          </Button>
        )}
      </Stack>
    );
  return <LoadedSettings {...props} view={config.data} />;
}

const choices = [
  { value: 'parakeet', label: 'Parakeet', description: 'MLX on Apple Silicon, sherpa-onnx elsewhere' },
  { value: 'parakeet-onnx', label: 'Parakeet ONNX', description: 'sherpa-onnx everywhere' },
  { value: 'whisper', label: 'Whisper', description: 'Needs the whisper service on' },
];

function Description({ text }: { text: string }): ReactNode {
  const parts: ReactNode[] = [];
  let start = 0;
  for (const match of text.matchAll(/\b[a-z_]{2,}\.[a-z_]{2,}(?:\.[a-z_]{2,})*\b|\[[a-z_.]{2,}\]/g)) {
    if (/\.(json|jsonl|wav|md|txt|sh|py|onnx|bin|jpg)$/.test(match[0])) continue;
    parts.push(text.slice(start, match.index), <Code key={`config-${match.index}`}>{match[0]}</Code>);
    start = match.index + match[0].length;
  }
  parts.push(text.slice(start));
  return parts;
}

function Field({
  row,
  value,
  values,
  disabled,
  onChange,
  error,
}: {
  row: Setting;
  value: string | boolean | undefined;
  values: Values;
  disabled: boolean;
  onChange: (value: string | boolean) => void;
  error?: string;
}) {
  const id = `set-${row.key}`;
  const label = (
    <>
      {nice(row.key.split('.').at(-1) ?? row.key)} <Code>{row.key}</Code>
    </>
  );
  const description = (
    <>
      <Description text={row.description} />
      {row.env && (
        <Text component="span" display="block" size="xs">
          {row.env} is set in the environment and overrides this.
        </Text>
      )}
    </>
  );
  const common = { id, disabled, error, 'aria-labelledby': `${id}-label`, 'aria-describedby': `${id}-help` };
  let control: ReactNode;
  if (row.type === 'bool')
    control = <Checkbox {...common} checked={Boolean(value)} onChange={e => onChange(e.currentTarget.checked)} />;
  else if (row.type === 'list')
    control = (
      <TagsInput
        {...common}
        value={String(value ?? '')
          .split(',')
          .map(s => s.trim())
          .filter(Boolean)}
        onChange={list => onChange(list.join(', '))}
        splitChars={[',']}
        placeholder="Type and press Enter"
      />
    );
  else if (row.key === 'asr.engine') {
    const options =
      choices.some(c => c.value === value) || !value
        ? choices
        : [...choices, { value: String(value), label: String(value), description: 'Current value' }];
    control = (
      <Radio.Group {...common} value={String(value ?? '')} onChange={onChange}>
        <Stack gap="xs" mt="xs">
          {options.map(c => (
            <Radio key={c.value} {...c} disabled={disabled} />
          ))}
        </Stack>
      </Radio.Group>
    );
  } else if (row.secret)
    control = (
      <PasswordInput
        {...common}
        value={String(value ?? '')}
        onChange={e => onChange(e.currentTarget.value)}
        autoComplete="off"
        placeholder={row.set ? 'Set. Leave blank to keep it.' : 'Not set'}
      />
    );
  else
    control = (
      <TextInput
        {...common}
        type={row.type === 'int' || row.type === 'float' ? 'number' : 'text'}
        step={row.type === 'float' ? 'any' : undefined}
        required={isRequired(row)}
        value={String(value ?? '')}
        onChange={e => onChange(e.currentTarget.value)}
        placeholder={
          row.key === 'owner.speaker_label'
            ? String(values['owner.name'] ?? '')
                .trim()
                .toLowerCase()
            : undefined
        }
      />
    );
  return (
    <SettingsRow className={classes.row} data-toggle={row.type === 'bool' || undefined}>
      <div className={layout.words}>
        <label id={`${id}-label`} htmlFor={id} className={layout.label}>
          {label}
        </label>
        <Text size="sm" c="dimmed" id={`${id}-help`}>
          {description}
        </Text>
      </div>
      <Stack gap="xs">
        {control}
        {row.key === 'owner.voice_ref' && <VoiceRef readOnly={disabled} />}
      </Stack>
    </SettingsRow>
  );
}

function LoadedSettings({
  view,
  setup = false,
  onClose,
  closeRequest = 0,
  onDirtyChange,
  onBusyChange,
}: SettingsFormProps & { view: ConfigView }) {
  const client = useQueryClient();
  const [saved] = useState(() => initialValues(view));
  const [values, setValues] = useState<Values>(saved);
  const [note, setNote] = useState('');
  const [failure, setFailure] = useState<{ error: string; field?: string } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const confirmId = useId();
  const confirmOpen = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (confirmOpen.current) modals.close(confirmId);
    };
  }, [confirmId]);
  useEffect(() => () => controller.current?.abort(), []);
  const changed = Object.keys(values).some(key => values[key] !== saved[key]);
  const rows = view.sections.flatMap(section => section.settings);
  const missing = rows.filter(row => isRequired(row) && blank(values[row.key]));
  const features = rows.filter(row => row.feature && row.type === 'bool');
  const [marked, setMarked] = useState('');
  // A field to open and focus: from a link, a refused save, or the browser's own validation (a new object each time).
  const [pending, setPending] = useState<{ id: string } | null>(null);
  const mutation = useMutation({
    mutationFn: async () => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      const { boot } = await request<{ boot: string }>('/health', undefined, abort.signal);
      const reply = await request<ConfigReply>('/config', { values }, abort.signal);
      if (!reply.ok) throw new HttpError(reply.error ?? 'Saving failed', 400, reply);
      await client.invalidateQueries({ queryKey: CONFIG_KEY, refetchType: 'none' });
      const overrides = reply.overridden?.length
        ? ` ${reply.overridden.join(', ')} stay as the environment sets them.`
        : '';
      setNote(`Saved to ${view.path}. Restarting with the new settings.${overrides}`);
      const url = restartDestination(reply.url);
      await followRestart(boot, url, abort.signal);
      if (!abort.signal.aborted) location.assign(url);
    },
    onError: error => {
      if (controller.current?.signal.aborted) return;
      const data = error instanceof HttpError ? error.data : null;
      const field =
        data && typeof data === 'object' && 'field' in data && typeof data.field === 'string' ? data.field : undefined;
      setNote('');
      setFailure({ error: messageOf(error), field });
    },
  });
  const busy = mutation.isPending;
  useEffect(() => {
    if (!busy && failure?.field) setPending({ id: `set-${failure.field}` }); // opens its section, then focuses it
  }, [busy, failure]);
  useEffect(() => onDirtyChange?.(changed), [changed, onDirtyChange]);
  useEffect(() => onBusyChange?.(busy), [busy, onBusyChange]);
  const close = () => {
    if (busy || !onClose) return;
    if (!changed) onClose();
    else if (!confirmOpen.current) {
      confirmOpen.current = true;
      void confirm({
        id: confirmId,
        title: 'Unsaved changes',
        body: 'You have unsaved changes. Discard them and close?',
        confirm: 'Discard',
        cancel: 'Keep editing',
        danger: true, // it throws the changes away: Keep editing is the safe default
      }).then(yes => {
        confirmOpen.current = false;
        if (yes && mounted.current) onClose();
      });
    }
  };
  const lastCloseRequest = useRef(closeRequest);
  useEffect(() => {
    if (lastCloseRequest.current === closeRequest) return;
    lastCloseRequest.current = closeRequest;
    close();
  });
  // Which sections are open: remembered per browser, plus any with an unsaved change or something to fix.
  const [open, setOpen] = useState<string[]>(loadOpen);
  useEffect(() => saveOpen(open), [open]);
  // A feature switch lives under New features (its own section only links to it), so it marks only that item.
  const states = new Map<string, SectionState>(
    view.sections.map(section => [
      section.name,
      sectionState(
        { ...section, settings: section.settings.filter(row => !(row.feature && row.type === 'bool')) },
        values,
        saved,
        failure?.field,
      ),
    ]),
  );
  const featureState = sectionState(
    { name: FEATURES, description: '', settings: features },
    values,
    saved,
    failure?.field,
  );
  states.set(FEATURES, featureState);
  const needed = [...states].filter(([, st]) => st.changed || st.trouble).map(([name]) => name);
  const neededKey = needed.join('\n');
  const wasNeeded = useRef<string[]>([]);
  // Only a section that has just come to need attention opens, so one the user closed stays closed.
  // biome-ignore lint/correctness/useExhaustiveDependencies: neededKey is needed, as text, so only a change reruns it
  useEffect(() => {
    const fresh = needed.filter(name => !wasNeeded.current.includes(name));
    wasNeeded.current = needed;
    if (fresh.length) setOpen(current => [...current, ...fresh.filter(name => !current.includes(name))]);
  }, [neededKey]);
  // A link to a field (#set-<key>) opens its section, then focuses the field once it shows.
  useEffect(() => {
    const focusHash = () => {
      let id: string;
      try {
        id = decodeURIComponent(location.hash.slice(1));
      } catch {
        return;
      }
      if (!id.startsWith('set-')) return;
      const item = itemOf(id, view.sections);
      if (item) setOpen(current => (current.includes(item) ? current : [...current, item]));
      setPending({ id });
    };
    focusHash();
    window.addEventListener('hashchange', focusHash);
    return () => window.removeEventListener('hashchange', focusHash);
  }, [view.sections]);
  useEffect(() => {
    if (!pending) return;
    const el = document.getElementById(pending.id);
    if (!el) return;
    const item = itemOf(pending.id, view.sections);
    if (item) setOpen(current => (current.includes(item) ? current : [...current, item]));
    setMarked(pending.id);
    // Its section may be opening: a field can't take focus until it shows, so try each frame for about half a second.
    let frame = 0;
    let tries = 30;
    const tryFocus = () => {
      el.scrollIntoView?.({ block: 'center' });
      el.focus();
      // A disabled field (read only, or while saving) never takes focus: scroll to it once and stop.
      if (document.activeElement !== el && !el.matches(':disabled') && --tries > 0)
        frame = requestAnimationFrame(tryFocus);
    };
    tryFocus();
    return () => cancelAnimationFrame(frame);
  }, [pending, view.sections]);
  const patch = (next: Values) => setValues(current => ({ ...current, ...next }));
  const save = (event: FormEvent) => {
    event.preventDefault();
    if (!view.control || busy || missing.length) return;
    setFailure(null);
    setNote('Saving');
    mutation.mutate();
  };
  return (
    <form
      onSubmit={save}
      // A field the browser finds invalid may sit in a closed section, where its message can't show: open and focus it.
      onInvalidCapture={event => {
        const id = (event.target as HTMLElement).id;
        if (id.startsWith('set-')) setPending({ id });
      }}
      aria-label={setup ? 'Set up Liberational' : 'Settings'}
    >
      <Stack>
        {setup && <Title order={1}>Set up Liberational</Title>}
        <Text>
          {setup ? (
            <>
              There is no <Code>{view.path}</Code> yet. Check these settings, then save. The dashboard restarts with
              them.
            </>
          ) : (
            <>
              Saved to <Code>{view.path}</Code>. Saving restarts the dashboard and any installed services.
            </>
          )}
        </Text>
        {!view.control && (
          <Alert color="yellow" title="Read only">
            Settings can only be changed from the machine Liberational runs on.
          </Alert>
        )}
        <Accordion
          multiple
          value={open}
          onChange={setOpen}
          order={3}
          keepMountedMode="display-none"
          variant="separated"
          classNames={{ item: layout.card, content: layout.panel }}
        >
          {!!features.length && (
            <Accordion.Item value={FEATURES} id="new-features">
              <SectionControl title="New features" count={features.length} state={featureState} />
              <Accordion.Panel>
                <Fieldset variant="unstyled" disabled={!view.control || busy}>
                  <div>
                    <SettingsRow>
                      <Text size="sm">
                        Newer ways of answering, each with its own switch. Uncheck one to go back to how it worked
                        before. Saving restarts, like the rest of Settings.
                      </Text>
                    </SettingsRow>
                    {features.map(row => (
                      <SettingsRow
                        key={row.key}
                        className={classes.row}
                        data-toggle
                        style={
                          marked === `set-${row.key}`
                            ? { outline: '2px solid var(--mantine-color-anchor)', outlineOffset: -2 }
                            : undefined
                        }
                      >
                        <div className={layout.words}>
                          <label id={`set-${row.key}-label`} htmlFor={`set-${row.key}`} className={layout.label}>
                            {row.feature}
                          </label>
                          <Text id={`set-${row.key}-help`} size="sm" c="dimmed">
                            <Description text={row.description} />{' '}
                            {row.default ? 'On by default.' : 'Off by default. Check it to opt in.'}
                          </Text>
                          <Code>{row.key}</Code>
                          {row.env && <Text size="xs">{row.env} is set in the environment and overrides this.</Text>}
                        </div>
                        <Checkbox
                          id={`set-${row.key}`}
                          aria-labelledby={`set-${row.key}-label`}
                          aria-describedby={`set-${row.key}-help`}
                          checked={Boolean(values[row.key])}
                          disabled={!view.control || busy}
                          onChange={e => patch({ [row.key]: e.currentTarget.checked })}
                        />
                      </SettingsRow>
                    ))}
                  </div>
                </Fieldset>
              </Accordion.Panel>
            </Accordion.Item>
          )}
          {view.sections.map(section => (
            <Accordion.Item key={section.name} value={section.name}>
              <SectionControl
                title={nice(section.name)}
                count={section.settings.filter(row => row.key !== 'services.disable').length}
                state={states.get(section.name)}
              />
              <Accordion.Panel>
                <Fieldset variant="unstyled" disabled={!view.control || busy}>
                  <div>
                    {section.description && (
                      <SettingsRow>
                        <Text size="sm">
                          <Description text={section.description} />
                        </Text>
                      </SettingsRow>
                    )}
                    {section.settings.map(row =>
                      row.key === 'services.disable' ? null : row.feature && row.type === 'bool' ? (
                        <Text key={row.key}>
                          <Code>{row.key}</Code>{' '}
                          <Anchor href={`#set-${row.key}`}>{row.feature}: on or off under New features</Anchor>
                        </Text>
                      ) : row.key === 'services.enable' ? (
                        <ServiceChecks
                          key={row.key}
                          values={values}
                          onChange={patch}
                          readOnly={!view.control || busy}
                        />
                      ) : (
                        <Field
                          key={row.key}
                          row={row}
                          value={values[row.key]}
                          values={values}
                          disabled={!view.control || busy}
                          onChange={value => patch({ [row.key]: value })}
                          error={failure?.field === row.key ? failure.error : undefined}
                        />
                      ),
                    )}
                  </div>
                </Fieldset>
              </Accordion.Panel>
            </Accordion.Item>
          ))}
        </Accordion>
        <Stack
          gap="xs"
          style={{
            position: 'sticky',
            bottom: 0,
            // The surface it sits on: the page for first-run setup, the drawer's card otherwise.
            background: setup ? 'var(--mantine-color-body)' : 'var(--lib-card)',
            paddingBlock: 12,
            zIndex: 1,
          }}
        >
          {failure && (
            <PageError title="Your changes weren't saved">
              {failure.field ? `${nice(failure.field)}: ` : ''}
              {failure.error}
            </PageError>
          )}
          {note && <Text role="status">{note}</Text>}
          {!!missing.length && !note && (
            <Text size="sm">Fill in {missing.map(row => nice(row.key)).join(', ')} to continue.</Text>
          )}
          <Group>
            {!setup && onClose && (
              <Button variant="default" disabled={busy} onClick={close}>
                Close
              </Button>
            )}
            <Button type="submit" disabled={!view.control || busy || !!missing.length}>
              {setup ? 'Save and start' : 'Save and restart'}
            </Button>
          </Group>
        </Stack>
      </Stack>
    </form>
  );
}

/** A section's header: its title, how many settings it holds, and a marker for an unsaved change or a problem. */
function SectionControl({ title, count, state }: { title: string; count: number; state?: SectionState }) {
  return (
    <Accordion.Control className={layout.accordionControl}>
      <Group gap="xs" wrap="nowrap">
        <Text fw={600} size="md">
          {title}
        </Text>
        <Text size="sm" c="dimmed">
          {count === 1 ? '1 setting' : `${count} settings`}
        </Text>
        {state?.trouble && (
          <Badge color="red" variant="light" size="sm">
            Needs attention
          </Badge>
        )}
        {state?.changed && !state.trouble && (
          <Badge color="yellow" variant="light" size="sm">
            Unsaved
          </Badge>
        )}
      </Group>
    </Accordion.Control>
  );
}
