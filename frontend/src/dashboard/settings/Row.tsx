// One setting in Settings (#382): its plain label and one line of help, the control, and Reset while it differs from
// what is saved, which also marks it as changed. In Advanced, its raw key and comment, and Default or Changed.
import { Button, Code, PasswordInput, Radio, Stack, Switch, TagsInput, Text, TextInput } from '@mantine/core';
import type { ReactNode } from 'react';
import { CONTROLS } from './controls';
import { useDraft } from './draft';
import { SettingsRow } from './layout';
import layout from './layout.module.css';
import { helpOf, isRequired, labelOf, listText, type Setting } from './model';
import classes from './settings.module.css';

const ENGINES = [
  { value: 'parakeet', label: 'Parakeet', description: 'Fast and accurate. The default.' },
  { value: 'parakeet-onnx', label: 'Parakeet (ONNX)', description: 'The same model on any CPU.' },
  { value: 'whisper', label: 'Whisper', description: 'Starts the whisper service.' },
];

/** A setting with a control of its own (controls.tsx) in the section that shows it. Advanced shows the raw value. */
const custom = (row: Setting) => (row.section ? CONTROLS[row.key] : undefined);
/** One whose control is not a field, so its label names it rather than pointing at an input. */
const notAField = (row: Setting) => row.section === 'you' && row.key === 'owner.voice_ref';

/** What an empty list asks for: for the wake words, the phrase used when there are none ("hey bob"). */
const wakePlaceholder = (row: Setting, name: unknown): string => {
  const n = String(name ?? '')
    .trim()
    .toLowerCase();
  return row.key === 'assistant.wake_words' && n ? `hey ${n}` : 'Type and press Enter';
};

/** The id a link to this setting focuses (/settings#set-<key>). */
export const fieldId = (key: string) => `set-${key}`;

function Control({ row, labelled }: { row: Setting; labelled: string }) {
  const d = useDraft();
  const value = d.value(row.key);
  const disabled = !d.view.control || d.busy;
  const id = fieldId(row.key);
  const changed = d.changed(row.key);
  const error = d.failure?.field === row.key ? d.failure.error : undefined;
  const common = {
    id,
    disabled,
    error,
    'aria-labelledby': labelled,
    classNames: { input: changed ? classes.changed : undefined },
  };
  const Own = custom(row);
  if (Own)
    return (
      <Own
        row={row}
        id={id}
        labelled={labelled}
        disabled={disabled}
        error={error}
        inputClass={changed ? classes.changed : undefined}
      />
    );
  if (row.type === 'bool')
    return (
      <Switch
        id={id}
        disabled={disabled}
        aria-labelledby={labelled}
        checked={Boolean(value)}
        onChange={e => d.set(row.key, e.currentTarget.checked)}
      />
    );
  if (row.type === 'list')
    return (
      <TagsInput
        {...common}
        value={String(value ?? '')
          .split(',')
          .map(s => s.trim())
          .filter(Boolean)}
        onChange={list => d.set(row.key, list.join(', '))}
        splitChars={[',']}
        placeholder={wakePlaceholder(row, d.value('assistant.name'))}
      />
    );
  if (row.key === 'asr.engine') {
    const options =
      ENGINES.some(c => c.value === value) || !value
        ? ENGINES
        : [...ENGINES, { value: String(value), label: String(value), description: 'Current value' }];
    return (
      <Radio.Group id={id} aria-labelledby={labelled} value={String(value ?? '')} onChange={v => d.set(row.key, v)}>
        <Stack gap="xs">
          {options.map(c => (
            <Radio key={c.value} {...c} disabled={disabled} />
          ))}
        </Stack>
      </Radio.Group>
    );
  }
  if (row.secret)
    return (
      <PasswordInput
        {...common}
        value={String(value ?? '')}
        onChange={e => d.set(row.key, e.currentTarget.value)}
        autoComplete="off"
        placeholder={row.set ? 'Set. Leave blank to keep it.' : 'Not set'}
      />
    );
  return (
    <TextInput
      {...common}
      type={row.type === 'int' || row.type === 'float' ? 'number' : 'text'}
      step={row.type === 'float' ? 'any' : undefined}
      required={isRequired(row)}
      value={String(value ?? '')}
      onChange={e => d.set(row.key, e.currentTarget.value)}
    />
  );
}

function ResetButton({ row }: { row: Setting }) {
  const d = useDraft();
  if (!d.changed(row.key)) return null;
  return (
    <Button
      variant="default"
      size="compact-xs"
      onClick={() => d.reset(row.key)}
      disabled={d.busy}
      aria-label={`Reset ${labelOf(row)}`}
    >
      Reset
    </Button>
  );
}

function EnvNote({ row }: { row: Setting }): ReactNode {
  if (!row.env) return null;
  return (
    <Text component="span" size="xs" c="dimmed" display="block">
      {row.env} is set in the environment and overrides this.
    </Text>
  );
}

/** A setting in its section: the label and help on the left, the control and Reset on the right. */
export function SettingRow({ row }: { row: Setting }) {
  const d = useDraft();
  const labelled = `${fieldId(row.key)}-label`;
  return (
    <SettingsRow
      component="fieldset"
      className={classes.row}
      data-toggle={row.type === 'bool' || undefined}
      aria-labelledby={labelled}
      data-changed={d.changed(row.key) || undefined}
    >
      <div className={layout.words}>
        {notAField(row) ? (
          <span id={labelled} className={layout.label}>
            {labelOf(row)}
          </span>
        ) : (
          <label id={labelled} htmlFor={fieldId(row.key)} className={layout.label}>
            {labelOf(row)}
          </label>
        )}
        <Text size="sm" c="dimmed">
          {helpOf(row)}
        </Text>
        {/* A feature says so only when it's not as it comes (#2728), instead of "On by default." on every line */}
        {row.feature && row.type === 'bool' && !sameAsDefault(row, d.value(row.key)) && (
          <Text size="xs" className={classes.isChanged}>
            Changed: {row.default ? 'on' : 'off'} by default
          </Text>
        )}
        <EnvNote row={row} />
      </div>
      <div className={classes.control}>
        <div className={classes.grow}>
          <Control row={row} labelled={labelled} />
        </div>
        <ResetButton row={row} />
      </div>
    </SettingsRow>
  );
}

/** A setting in Advanced: its raw key, control with description below, and compact default state. */
export function RawRow({ row }: { row: Setting }) {
  const d = useDraft();
  const labelled = `${fieldId(row.key)}-label`;
  const isDefault = sameAsDefault(row, d.value(row.key)); // as it would be saved, against defaults.toml
  return (
    <SettingsRow
      component="fieldset"
      className={classes.row}
      aria-labelledby={labelled}
      aria-describedby={`${fieldId(row.key)}-help`}
      data-changed={d.changed(row.key) || undefined}
      data-toggle={row.type === 'bool' || undefined}
    >
      <div className={layout.words}>
        <Code id={labelled} className={classes.key}>
          {row.key}
        </Code>
        <Text size="sm" c="dimmed" className={classes.desc} id={`${fieldId(row.key)}-help`}>
          {row.description}
          <EnvNote row={row} />
        </Text>
      </div>
      <div className={classes.control}>
        <div className={classes.grow}>
          <Control row={{ ...row, section: '' }} labelled={labelled} />
        </div>
        <ResetButton row={row} />
        <Text size="xs" className={isDefault ? classes.isDefault : classes.isChanged}>
          {isDefault ? 'Default' : 'Changed'}
        </Text>
      </div>
    </SettingsRow>
  );
}

/** Whether a setting's value is its default in defaults.toml (a list as typed, "a, b"). */
export function sameAsDefault(row: Setting, value: string | boolean | undefined): boolean {
  if (row.secret) return !row.set && !value; // a new secret typed in will be set
  if ((row.type === 'int' || row.type === 'float') && typeof value === 'string' && value.trim() !== '')
    return Number(value) === Number(row.default); // "30.0" is 30
  const def = Array.isArray(row.default)
    ? listText(row.default) // as the draft holds a list (JSON when an item has a comma)
    : row.type === 'bool'
      ? Boolean(row.default)
      : String(row.default ?? '');
  return value === def;
}
