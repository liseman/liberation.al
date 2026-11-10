// Settings' sources, moved here from the Ingest tab, which keeps uploads, links, and what was brought in. Each
// source is shown by what it is (model.ts): camera sources (Wyze) on Cameras and microphones' Network cameras card,
// and photo libraries and recorders (Google Photos, iCloud Photos, AllSound, Thinking Hat) under Imported media. Each
// row says whether the source is set up and has its own options. It reads the same GET /ingest/api/items the tab
// does (dashboard/ingest.py) and calls the same routes; nothing about how a source works changed.
import {
  Anchor,
  Badge,
  Box,
  Button,
  Checkbox,
  Group,
  Image,
  Loader,
  NativeSelect,
  PasswordInput,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type ChangeEvent, type FormEvent, type JSX, useState } from 'react';
import { messageOf } from '../../../shared/http';
import {
  type Album,
  getIngest,
  googleAlbums,
  googleImport,
  googleLogin,
  googlePhotos,
  INGEST_KEY,
  type IngestView,
  pollSource,
  type RemotePhoto,
  type SourceInfo,
  saveWyze,
  type WyzeSettings,
  wyzeValues,
} from '../../ingest/api';
import { type Message, muted, Status, useRefresh, when } from '../../ingest/bits';
import { plural } from '../../lib/format';
import { PageError } from '../../lib/page';
import { href, linkClick } from '../../router';
import { SettingsRow, SettingsSection, SourceItem } from '../layout';
import layout from '../layout.module.css';
import { CONFIG_KEY } from '../model';
import { CARD_ID, kindOf, type SourceKind } from './model';
import { NetworkCameraDiscovery } from './NetworkCameraDiscovery';
import type { CameraNamesDraft } from './networkCameras';
import classes from './sources.module.css';

function GooglePhotos({ source, control }: { source: SourceInfo; control: boolean }) {
  const refresh = useRefresh();
  const [msg, setMsg] = useState<Message>(null);
  const [album, setAlbum] = useState<string | null>(null);
  const [shown, setShown] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const albums = useQuery({
    queryKey: [...INGEST_KEY, 'google', 'albums'],
    queryFn: ({ signal }) => googleAlbums(signal),
    enabled: shown,
    retry: false,
    staleTime: 60_000,
  });
  const photos = useQuery({
    queryKey: [...INGEST_KEY, 'google', 'photos', album],
    queryFn: ({ signal }) => googlePhotos(album, signal),
    enabled: shown,
    retry: false,
    staleTime: 60_000,
  });
  const list: RemotePhoto[] = photos.data?.items ?? [];
  const toggle = (key: string) =>
    setPicked(p => {
      const next = new Set(p);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const bring = async () => {
    try {
      await googleImport(list.filter(p => picked.has(p.key)));
      setMsg({ text: `Importing ${plural(picked.size, 'item', 'items')} in the background.` });
      setPicked(new Set());
    } catch (e) {
      setMsg({ err: messageOf(e) });
    }
    void refresh();
  };
  return (
    <Stack gap="xs" mt="xs">
      <Group gap="xs">
        {control && (
          <Button
            size="xs"
            variant="default"
            // While it needs a package installed (the source says so), signing in can only fail
            disabled={!!source.blocked}
            onClick={async () => {
              try {
                await googleLogin();
                setMsg({ text: 'The sign-in window is open on this computer. Sign in, then close it.' });
              } catch (e) {
                setMsg({ err: messageOf(e) });
              }
            }}
          >
            Open sign-in window
          </Button>
        )}
        <Button size="xs" variant="default" disabled={!source.configured} onClick={() => setShown(true)}>
          Browse photos
        </Button>
      </Group>
      {shown && (
        <>
          <NativeSelect
            label="Album"
            value={album ?? ''}
            onChange={e => {
              setAlbum(e.currentTarget.value || null);
              setPicked(new Set());
            }}
            data={[
              { value: '', label: 'Newest in the library' },
              ...(albums.data?.albums ?? []).map((a: Album) => ({ value: a.id, label: a.title })),
            ]}
          />
          {albums.isError && <Status msg={{ err: messageOf(albums.error) }} />}
          {photos.isPending ? (
            <Group gap="xs" role="status" aria-live="polite">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                Loading photos
              </Text>
            </Group>
          ) : photos.isError ? (
            <Status msg={{ err: messageOf(photos.error) }} />
          ) : list.length === 0 ? (
            <Text {...muted}>No photos found here.</Text>
          ) : (
            <SimpleGrid cols={{ base: 3, sm: 6 }} spacing="xs">
              {list.map(p => (
                <Box key={p.key} className={classes.tile}>
                  <Image src={p.thumb} alt={p.label || p.name} radius="sm" />
                  <Checkbox
                    size="xs"
                    mt={4}
                    label={p.taken_at ? when(p.taken_at) : p.kind}
                    checked={picked.has(p.key)}
                    onChange={() => toggle(p.key)}
                  />
                </Box>
              ))}
            </SimpleGrid>
          )}
          <Group>
            <Button size="xs" variant="default" disabled={!picked.size} onClick={bring}>
              Import {picked.size || ''}
            </Button>
          </Group>
        </>
      )}
      <Status msg={msg} />
    </Stack>
  );
}

/** The Wyze bridge's sign-in and address. Folded behind a button: while it isn't set up, the whole form
 *  took the card's room above the cameras that work. */
function Wyze({ wyze, control, configured }: { wyze: WyzeSettings; control: boolean; configured: boolean }) {
  const [open, setOpen] = useState(false);
  if (!open)
    return (
      <Button size="xs" variant="default" mt="xs" onClick={() => setOpen(true)}>
        {configured ? 'Change Wyze settings' : 'Set up Wyze cameras'}
      </Button>
    );
  return <WyzeForm wyze={wyze} control={control} />;
}

function WyzeForm({ wyze, control }: { wyze: WyzeSettings; control: boolean }) {
  const refresh = useRefresh();
  const client = useQueryClient();
  const [msg, setMsg] = useState<Message>(null);
  const [form, setForm] = useState({
    cameras: wyze.cameras.join(', '),
    rtsp_url: wyze.rtsp_url,
    email: '',
    password: '',
    api_id: '',
    api_key: '',
  });
  const field = (k: keyof typeof form) => ({
    value: form[k],
    onChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      setForm({ ...form, [k]: e.currentTarget.value }),
    disabled: !control,
  });
  const saved = (set: boolean) => (set ? 'Saved; type to replace' : 'Not set');
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const reply = await saveWyze(wyzeValues(form));
      const restarted = reply.restarted?.length ? ` Restarted ${reply.restarted.join(', ')}.` : '';
      setMsg({ text: `Saved.${restarted}` });
      setForm({ ...form, email: '', password: '', api_id: '', api_key: '' });
      // These are settings too: Advanced shows them, and the draft keeps anything changed there (draft.tsx).
      void client.invalidateQueries({ queryKey: CONFIG_KEY });
    } catch (err) {
      setMsg({ err: messageOf(err) });
    }
    void refresh();
  };
  return (
    <Box component="form" onSubmit={submit} mt="xs">
      <Text {...muted}>
        Wyze cameras come in through docker-wyze-bridge (see docs/INGEST.md). The bridge signs in with these; each
        camera is recorded from its RTSP stream.
      </Text>
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="xs" mt="xs">
        <TextInput label="Wyze email" placeholder={saved(wyze.email)} {...field('email')} />
        <PasswordInput label="Wyze password" placeholder={saved(wyze.password)} {...field('password')} />
        <TextInput label="API ID" placeholder={saved(wyze.api_id)} {...field('api_id')} />
        <PasswordInput label="API key" placeholder={saved(wyze.api_key)} {...field('api_key')} />
        <TextInput label="Bridge RTSP address" {...field('rtsp_url')} />
        <Textarea
          label="Cameras"
          description="The bridge's name for each, separated by commas"
          rows={2}
          {...field('cameras')}
        />
      </SimpleGrid>
      {control ? (
        <Button type="submit" variant="default" size="xs" mt="xs">
          Save Wyze settings
        </Button>
      ) : (
        <Text {...muted} mt="xs">
          Wyze settings can only be changed from this computer.
        </Text>
      )}
      <Status msg={msg} />
    </Box>
  );
}

function Source({ source, view }: { source: SourceInfo; view: IngestView }) {
  const refresh = useRefresh();
  const [msg, setMsg] = useState<Message>(null);
  const check = async () => {
    try {
      await pollSource(source.name);
      setMsg({ text: 'Checking for new items.' });
    } catch (e) {
      setMsg({ err: messageOf(e) });
    }
    void refresh();
  };
  const options =
    source.name === 'google_photos' ? (
      <GooglePhotos source={source} control={view.control} />
    ) : source.name === 'wyze' ? (
      <Wyze wyze={view.wyze} control={view.control} configured={source.configured} />
    ) : null;
  // No on switch yet: a source is on once it's set up (ingest.sources picks which ongoing ones pack-ingest polls).
  return (
    <SourceItem
      component="li"
      aria-label={source.label}
      on
      head={
        <SettingsRow>
          <Group justify="space-between" wrap="nowrap">
            <Stack gap={0} style={{ minWidth: 0 }}>
              <Group gap="xs">
                <Text component="p" m={0} className={layout.label}>
                  {source.label}
                </Text>
                <Badge size="sm" variant="light" color={source.configured ? 'green' : 'gray'}>
                  {source.configured ? 'Ready' : 'Not set up'}
                </Badge>
                {source.ongoing && (
                  // Mantine's grey label read at 3.3:1 on white; the page's dimmed text reads at 4.5:1 or better
                  <Badge size="sm" variant="outline" color="gray" c="var(--mantine-color-dimmed)">
                    Ongoing
                  </Badge>
                )}
              </Group>
              <Text {...muted}>
                {source.message}
                {source.seen ? `. ${plural(source.seen, 'item', 'items')} brought in so far.` : ''}
              </Text>
            </Stack>
            {source.ongoing && source.configured && (
              <Button size="xs" variant="default" onClick={check}>
                Check now
              </Button>
            )}
          </Group>
          <Status msg={msg} />
        </SettingsRow>
      }
    >
      {options && <SettingsRow>{options}</SettingsRow>}
    </SourceItem>
  );
}

function useIngestView() {
  return useQuery({
    queryKey: INGEST_KEY,
    queryFn: ({ signal }) => getIngest(signal),
    refetchInterval: 30_000,
  });
}

/** A card of the sources of one kind, or what's wrong while they load. */
/** "A, B, and C". */
const listWords = (words: readonly string[]): string =>
  words.length < 3 ? words.join(' and ') : `${words.slice(0, -1).join(', ')}, and ${words.at(-1)}`;

/** The sources there's no way to bring in from yet, as one line, with where their exports go instead. */
function ComingLater({ sources }: { sources: readonly SourceInfo[] }) {
  return (
    <SettingsRow>
      <Text size="sm" c="dimmed">
        Coming later: {listWords(sources.map(s => s.label))}. For now,{' '}
        <Anchor href={href('/ingest')} onClick={linkClick} underline="always">
          upload their exports on Ingest
        </Anchor>
        .
      </Text>
    </SettingsRow>
  );
}

function SourcesCard({
  kind,
  title,
  description,
  empty,
  cameraNames,
}: {
  kind: Exclude<SourceKind, 'none'>;
  title: string;
  description: string;
  empty: string;
  cameraNames?: CameraNamesDraft;
}) {
  const query = useIngestView();
  const view = query.data;
  const all = view?.sources.filter(s => kindOf(s.name) === kind) ?? [];
  // What can be connected now comes first; sources with no way in yet are one line under it
  const shown = all.filter(s => !s.later);
  const later = all.filter(s => s.later);
  return (
    <SettingsSection title={title} description={description} id={CARD_ID[kind]} tabIndex={-1}>
      {view ? (
        all.length ? (
          <>
            <ul className={classes.list}>
              {shown.map(s => (
                <Source key={s.name} source={s} view={view} />
              ))}
            </ul>
            {later.length > 0 && <ComingLater sources={later} />}
          </>
        ) : (
          <SettingsRow>
            <Text c="dimmed">{empty}</Text>
          </SettingsRow>
        )
      ) : (
        <SettingsRow>
          {query.error ? (
            <PageError
              title="Couldn't load sources"
              detail={messageOf(query.error)}
              onRetry={() => void query.refetch()}
              busy={query.isFetching}
            />
          ) : (
            <Group gap="xs" role="status" aria-live="polite">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                Looking for sources…
              </Text>
            </Group>
          )}
        </SettingsRow>
      )}
      {kind === 'camera' && <NetworkCameraDiscovery names={cameraNames} />}
    </SettingsSection>
  );
}

/** Network cameras: Wyze ingest plus native RTSP configured/discovered through Protect and ONVIF. */
export function NetworkCameras({ names }: { names?: CameraNamesDraft } = {}): JSX.Element {
  return (
    <SourcesCard
      kind="camera"
      title="Network cameras"
      description="Cameras that aren't plugged into this machine, recorded over the network."
      empty="No network camera sources."
      cameraNames={names}
    />
  );
}

/** Photo libraries and recorders whose media is imported (Imported media). */
export function ImportedMedia(): JSX.Element {
  return (
    <SourcesCard
      kind="media"
      title="Accounts and apps"
      description="Photo libraries and recorders to bring photos, video, and audio in from. What they bring in is listed on the Ingest tab."
      empty="No sources to import from."
    />
  );
}
