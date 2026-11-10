// Settings as a page (#382): a menu of sections on the left with a search across every setting, the section on
// the right, and a save bar once something has changed. A link to a setting (/settings#set-<key>) opens its section
// and focuses it; a link to a section (/settings#<id>) opens it, and one to a card in it (an anchor, catalog.ts)
// opens it and scrolls to that card. At phone width the menu and the section take turns:
// the menu first, then the section with a link back to it (#787). First-run setup is SettingsForm's.
import { Alert, Button, NavLink, Text, TextInput, Title } from '@mantine/core';
import { IconChevronLeft } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { use, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { messageOf, request } from '../../shared/http';
import { Deferred } from '../Deferred';
import { PageError, PageLoading } from '../lib/page';
import { redirect } from '../router';
import { Advanced } from './Advanced';
import { homeOf, SECTIONS, type SectionDef, type SectionId, sectionInHash, sectionOf } from './catalog';
import { DEVICE_CARD } from './devices/cards';
import { DraftProvider, useDraft } from './draft';
import { SettingsHeader, SettingsSection } from './layout';
import { CONFIG_KEY, type ConfigView, labelOf, restartsFor, type Setting } from './model';
import { fieldId, RawRow, SettingRow } from './Row';
import { SearchSlot } from './searchSlot';
import classes from './settings.module.css';

export function SettingsPage() {
  const config = useQuery({
    queryKey: CONFIG_KEY,
    queryFn: ({ signal }) => request<ConfigView>('/config', undefined, signal),
  });
  if (!config.data)
    return (
      <div className={classes.loading}>
        {config.error ? (
          <PageError
            title="Couldn't load the settings"
            detail={messageOf(config.error)}
            onRetry={() => void config.refetch()}
            busy={config.isFetching}
          />
        ) : (
          <PageLoading label="settings" />
        )}
      </div>
    );
  return (
    <DraftProvider view={config.data}>
      <Page />
    </DraftProvider>
  );
}

/** Whether a setting matches a search: its words, its key, or its comment. */
export function matches(row: Setting, query: string): boolean {
  const q = query.trim().toLocaleLowerCase();
  if (!q) return false;
  return [labelOf(row), row.help, row.key, row.description, row.feature].some(t => t?.toLocaleLowerCase().includes(q));
}

function Page() {
  const d = useDraft();
  const sectionOfKey = (key: string) => d.all.find(r => r.key === key)?.section ?? '';
  const [current, setCurrent] = useState<SectionId>(() => sectionInHash(location.hash, sectionOfKey) ?? 'you');
  // At phone width, which of the two shows: a section's settings (the first, unless one is linked to), with Back to
  // the menu, so Settings opens on settings rather than a list of names.
  const [shown, setShown] = useState<'menu' | 'section'>('section');
  const [query, setQuery] = useState('');
  const [focus, setFocus] = useState<{ id: string } | null>(null); // a new object each time, so a repeat refocuses
  const open = (id: SectionId) => {
    setFocus(null);
    setCurrent(id);
    setShown('section');
    setQuery('');
    history.replaceState(history.state, '', `${location.pathname}${location.search}#${id}`);
  };
  // Links in: a section, or a setting, whose section opens and whose field takes focus.
  // biome-ignore lint/correctness/useExhaustiveDependencies: d.all only changes with the view, which remounts this
  useEffect(() => {
    const follow = () => {
      // Phones, watches, and tags moved to Known › Devices: an old link to their card goes there.
      if (location.hash === `#${DEVICE_CARD.links}`) {
        redirect('/devices');
        return;
      }
      const id = sectionInHash(location.hash, sectionOfKey);
      if (!id) return;
      setCurrent(id);
      setShown('section');
      setQuery('');
      // A setting's field, or a card in a section's panel (an anchor), takes focus; a bare section only opens.
      const target = decodeURIComponent(location.hash.slice(1));
      setFocus(target !== id ? { id: target } : null);
    };
    follow();
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);
  // A save the server refused names a field: open its section and focus it, once the save has stopped (a field
  // can't take focus while saving disables it).
  // biome-ignore lint/correctness/useExhaustiveDependencies: only a new failure, or the save stopping, moves the focus
  useEffect(() => {
    const field = d.failure?.field;
    if (!field || d.busy) return;
    setCurrent(homeOf(field, sectionOfKey));
    setShown('section');
    setQuery('');
    setFocus({ id: fieldId(field) });
  }, [d.failure, d.busy]);
  useEffect(() => {
    if (!focus) return;
    let frame = 0;
    const observer = new MutationObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(tryFocus);
    });
    function tryFocus() {
      const el = document.getElementById(focus?.id ?? '');
      if (!el) return;
      // A field is centered; a card (an anchor, like #device-links) shows from its heading down, since a tall one
      // centered would hide its title.
      el.scrollIntoView?.({ block: focus?.id.startsWith('set-') ? 'center' : 'start' });
      el.focus({ preventScroll: true });
      if (document.activeElement === el || el.matches(':disabled')) observer.disconnect();
    }
    // Deferred sections and their API data can arrive after a slow download. Observe their commit instead of
    // giving up after a fixed number of animation frames, and stop observing as soon as the target is ready.
    observer.observe(document.body, { childList: true, subtree: true });
    tryFocus();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [focus]);
  const found = useMemo(() => d.all.filter(row => matches(row, query)), [d.all, query]);
  // In the drawer, the search sits in its header row; first-run setup keeps it atop the menu.
  const slot = use(SearchSlot);
  const search = (
    <TextInput
      type="search"
      aria-label="Search settings"
      placeholder="Search settings"
      value={query}
      onChange={e => {
        setFocus(null);
        setQuery(e.currentTarget.value);
      }}
      onKeyDown={e => e.key === 'Escape' && setQuery('')}
      className={slot ? undefined : classes.search}
    />
  );
  const searching = query.trim() !== '';
  const section = sectionOf(current);
  return (
    <div className={classes.page} data-shown={shown} data-searching={searching || undefined}>
      {slot && createPortal(search, slot)}
      <nav aria-label="Settings sections" className={classes.nav}>
        <Title order={1} className={classes.title}>
          Settings
        </Title>
        {!slot && search}
        {SECTIONS.map(s => (
          <NavItem key={s.id} section={s} current={!searching && s.id === current} onOpen={() => open(s.id)} />
        ))}
      </nav>
      <div className={classes.main}>
        {!searching && (
          <Button
            variant="subtle"
            size="compact-md"
            className={classes.back}
            leftSection={<IconChevronLeft size={16} aria-hidden />}
            aria-label="Back to Settings"
            onClick={() => setShown('menu')}
          >
            Settings
          </Button>
        )}
        {!d.view.control && (
          <Alert color="gray" title="Read only" className={classes.readOnly}>
            Settings can be changed only from the machine Liberational runs on.
          </Alert>
        )}
        {searching ? <Results query={query} found={found} /> : <SectionView section={section} />}
        <SaveBar />
      </div>
    </div>
  );
}

function NavItem({ section, current, onOpen }: { section: SectionDef; current: boolean; onOpen: () => void }) {
  const d = useDraft();
  const own = section.useBadge?.() ?? null; // a section's own badge, from its own data (S3, S5)
  const rows = d.rows(section.id);
  const trouble = section.id !== 'advanced' && rows.some(r => d.missing.includes(r) || d.failure?.field === r.key);
  const unsaved = section.id !== 'advanced' ? rows.filter(r => d.changed(r.key)).length : 0;
  const badge = trouble
    ? { text: 'Needs attention', tone: 'warn' as const }
    : (own ?? (unsaved ? { text: `${unsaved} unsaved`, tone: 'dim' as const } : null));
  return (
    <NavLink
      href={`#${section.id}`}
      active={current}
      label={section.title}
      rightSection={badge ? <span className={classes[badge.tone]}>{badge.text}</span> : undefined}
      classNames={{ label: classes.navLabel }}
      aria-current={current ? 'page' : undefined}
      className={classes.navItem}
      onClick={e => {
        e.preventDefault();
        onOpen();
      }}
    />
  );
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-');

/** A section's rows by card (#385): each group in the order its first row comes. Rows with no group share one
 *  untitled card, or, in a section with titled groups, come last under "Other" (a new feature not yet in
 *  labels.toml lands there). */
export function groups(rows: Setting[]): [string, Setting[]][] {
  const byGroup = new Map<string, Setting[]>();
  for (const row of rows) byGroup.set(row.group ?? '', [...(byGroup.get(row.group ?? '') ?? []), row]);
  const loose = byGroup.get('');
  byGroup.delete('');
  if (!byGroup.size) return loose ? [['', loose]] : [];
  if (loose) byGroup.set('Other', [...(byGroup.get('Other') ?? []), ...loose]); // one Other, even if labels.toml has one
  return [...byGroup];
}

function SectionView({ section }: { section: SectionDef }) {
  const d = useDraft();
  const rows = d
    .rows(section.id)
    .filter(row => !section.drawn?.includes(row.key) && !section.hidden?.includes(row.key));
  const Body = section.Body;
  const raw = section.id === 'advanced';
  return (
    <section aria-labelledby="settings-section-h" className={classes.section}>
      <SettingsHeader title={section.title} description={section.blurb} id="settings-section-h" />
      {Body && (
        <Deferred key={section.id} name={section.title}>
          <Body />
        </Deferred>
      )}
      {rows.length ? (
        raw ? (
          <Advanced />
        ) : (
          groups(rows).map(([group, inGroup]) => (
            <SettingsSection
              key={group}
              title={group || undefined}
              headingId={group ? `settings-group-${slug(group)}` : undefined}
            >
              {inGroup.map(row => (
                <SettingRow key={row.key} row={row} />
              ))}
            </SettingsSection>
          ))
        )
      ) : (
        !Body && (
          <Text c="dimmed" className={classes.empty}>
            Nothing to set here yet. Every setting is under Advanced.
          </Text>
        )
      )}
    </section>
  );
}

function Results({ query, found }: { query: string; found: Setting[] }) {
  return (
    <section aria-label="Search results" className={classes.section}>
      <SettingsHeader
        title={
          found.length ? `${found.length} ${found.length === 1 ? 'setting' : 'settings'} match` : 'No setting matches'
        }
        description={found.length ? `Searching for "${query.trim()}"` : 'Try other words, or look under Advanced.'}
      />
      {SECTIONS.filter(s => s.id !== 'advanced').map(s => {
        const here = found.filter(r => r.section === s.id);
        return here.length ? (
          <SettingsSection key={s.id} title={s.title}>
            {here.map(row => (
              <SettingRow key={row.key} row={row} />
            ))}
          </SettingsSection>
        ) : null;
      })}
      {found.some(r => !r.section) && (
        <SettingsSection title="Advanced">
          {found
            .filter(r => !r.section)
            .map(row => (
              <RawRow key={row.key} row={row} />
            ))}
        </SettingsSection>
      )}
    </section>
  );
}

function SaveBar() {
  const d = useDraft();
  const n = d.changedKeys.length;
  if (!n && !d.note && !d.failure) return null;
  const names = restartsFor(d.view, d.changedKeys);
  const restarts = names?.length
    ? ` Saving restarts: ${names.join(', ')}`
    : names && n
      ? ' Saving restarts nothing.'
      : '';
  const blocked = d.missing.length ? `Fill in ${d.missing.map(labelOf).join(', ')} to save.` : '';
  return (
    <section className={classes.bar} aria-label="Unsaved changes">
      <div className={classes.barText}>
        {d.failure ? (
          <Text size="sm" role="alert" className={classes.barError}>
            {d.failure.error}
          </Text>
        ) : d.note ? (
          <Text size="sm" role="status">
            {d.note}
          </Text>
        ) : (
          <Text size="sm" role="status">
            {`${n} unsaved ${n === 1 ? 'change' : 'changes'}.${restarts}`}
            {blocked && <span className={classes.barBlocked}> {blocked}</span>}
          </Text>
        )}
      </div>
      <div className={classes.barActions}>
        <Button variant="default" className={classes.barDiscard} onClick={d.discard} disabled={d.busy}>
          Discard
        </Button>
        <Button
          className={classes.barSave}
          onClick={d.save}
          loading={d.busy}
          disabled={!n || !!d.missing.length || !d.view.control}
        >
          {names && !names.length ? 'Save' : 'Save and restart'}
        </Button>
      </div>
    </section>
  );
}
