// Advanced: every setting by its raw key and its defaults.toml comment, a card per config section, with a
// filter and "Only changed from default", as the design draws it (Advanced.dc.html).
import { Button, Checkbox, Group, Text, TextInput } from '@mantine/core';
import { useEffect, useState } from 'react';
import { useDraft } from './draft';
import { SettingsSection } from './layout';
import { labelOf, nice, type Section, type Setting } from './model';
import { RawRow, sameAsDefault } from './Row';
import classes from './settings.module.css';

/** A config section's title in words: "presence" is "Presence", "models.files.vad_silero" is "Models / Files / Vad
 *  silero". */
export const sectionTitle = (name: string): string => name.split('.').map(nice).join(' / ');

/** The first sentence of a section's comment, as the card's note (the full comment is long), cut at a word near
 *  NOTE_MAX characters when even that runs on. */
export const NOTE_MAX = 110;
export function sectionNote(description: string): string {
  const text = description.replace(/\s+/g, ' ').trim();
  // A sentence ends at . ! or ? before a space or the end, but not in "e.g." or "i.e." or after "vs" or "etc"
  const end = [...text.matchAll(/[.!?](?=\s|$)/g)].find(
    m => !/\b(e\.g|i\.e|vs|etc)$/i.test(text.slice(0, m.index)),
  )?.index;
  const first = end == null ? text : text.slice(0, end + 1);
  if (first.length <= NOTE_MAX) return first;
  let cut = first.slice(0, NOTE_MAX);
  cut = cut.slice(0, cut.lastIndexOf(' ') > 0 ? cut.lastIndexOf(' ') : NOTE_MAX);
  const open = cut.lastIndexOf('(');
  if (open > cut.lastIndexOf(')')) cut = cut.slice(0, open); // not "(face visits in...": drop the aside it cut into
  return `${cut.trim().replace(/[,;:]+$/, '')}\u2026`;
}

/** Whether a filter (lowercased) finds a setting: by its key, its comment, its plain label and help, or the new
 *  feature it switches. */
export const matchesFilter = (row: Setting, q: string): boolean =>
  [row.key, row.description, labelOf(row), row.help, row.feature].some(s => s?.toLowerCase().includes(q));

/** The setting a #set-<key> link names, or null. */
const linkedKey = (): string | null =>
  location.hash.startsWith('#set-') ? decodeURIComponent(location.hash.slice('#set-'.length)) : null;

export function Advanced() {
  const d = useDraft();
  const [filter, setFilter] = useState('');
  const [changedOnly, setChangedOnly] = useState(false);
  // Each config section folds to its heading and count: the filter comes first,
  // and a section opens when the owner opens it, a filter or "Only changed" is on, or it holds a linked, edited, or
  // refused setting. On a phone, every key listed in full ran to 32,000 px.
  const [opened, setOpened] = useState<ReadonlySet<string>>(() => new Set());
  const [linked, setLinked] = useState(linkedKey);
  // A link to one setting (#set-<key>) must find it: drop a filter that would hide it, and open its section
  useEffect(() => {
    const show = () => {
      if (!location.hash.startsWith('#set-')) return;
      setFilter('');
      setChangedOnly(false);
      setLinked(linkedKey());
    };
    addEventListener('hashchange', show);
    return () => removeEventListener('hashchange', show);
  }, []);
  const q = filter.trim().toLowerCase();
  // A row being edited, or one a save refused, always shows: hiding it would drop the field mid-edit, or leave the
  // refused one nowhere to focus
  const keep = (row: Setting) =>
    d.changed(row.key) ||
    d.failure?.field === row.key ||
    ((!q || matchesFilter(row, q)) && (!changedOnly || !sameAsDefault(row, d.value(row.key))));
  const shown: [Section, Setting[]][] = d.view.sections
    .map(s => [s, s.settings.filter(keep)] as [Section, Setting[]])
    .filter(([, rows]) => rows.length);
  return (
    <>
      <Group gap="lg" align="center" wrap="wrap">
        <TextInput
          label="Filter settings"
          placeholder="By key or words, like gap"
          value={filter}
          onChange={e => setFilter(e.currentTarget.value)}
          className={classes.filter}
        />
        <Checkbox
          label="Only changed from default"
          checked={changedOnly}
          onChange={e => setChangedOnly(e.currentTarget.checked)}
          mt="lg"
        />
      </Group>
      {shown.length ? (
        shown.map(([section, rows]) => {
          const id = `advanced-${section.name.replace(/[^a-z0-9]+/gi, '-')}`;
          const note = sectionNote(section.description);
          const searching = !!q || changedOnly;
          const held = rows.some(r => r.key === linked || d.changed(r.key) || d.failure?.field === r.key);
          const open = searching || held || opened.has(section.name);
          const toggle = () =>
            setOpened(was => {
              const next = new Set(was);
              if (next.has(section.name)) next.delete(section.name);
              else next.add(section.name);
              return next;
            });
          return (
            <SettingsSection
              key={section.name}
              headingId={id}
              title={sectionTitle(section.name)}
              description={note || undefined}
              trailing={
                <span className={classes.advancedTrail}>
                  <span className={classes.dimCount}>
                    {rows.length === 1 ? '1 setting' : `${rows.length} settings`}
                  </span>
                  {!searching && !held && (
                    <Button
                      size="compact-xs"
                      variant="subtle"
                      aria-expanded={open}
                      aria-controls={`${id}-rows`}
                      aria-label={`${open ? 'Hide' : 'Show'} ${sectionTitle(section.name)}`}
                      onClick={toggle}
                    >
                      {open ? 'Hide' : 'Show'}
                    </Button>
                  )}
                </span>
              }
            >
              <div id={`${id}-rows`} hidden={!open}>
                {open && rows.map(row => <RawRow key={row.key} row={row} />)}
              </div>
            </SettingsSection>
          );
        })
      ) : (
        <Text c="dimmed" className={classes.empty} role="status">
          {changedOnly && !q
            ? 'Every setting is at its default.'
            : `No setting matches${changedOnly ? ' among those changed from default' : ''}.`}
        </Text>
      )}
    </>
  );
}
