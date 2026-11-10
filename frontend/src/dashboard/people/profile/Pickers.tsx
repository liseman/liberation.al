// "Pick what's theirs": after New person (or from a card's More menu), the faces, video, and voices nobody has named
// yet, to pick from and give to this person in one go (POST /people/assign, #309).
import { Button, Group, Switch, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type ReactNode, useState } from 'react';
import { HttpError } from '../../../shared/http';
import { ProcessingImages } from '../../../shared/ProcessingImages';
import { SafeImage } from '../../../shared/SafeImage';
import { ClipPoster } from '../../clip/ClipPoster';
import { clipLength, plural } from '../../lib/format';
import { BlockSkeleton, useSlow } from '../../lib/loading';
import { PlayClip } from '../../lib/PlayToggle';
import { LoadError } from '../../lib/page';
import { type GridTile, SelectableGrid } from '../../lib/SelectableGrid';
import {
  type AssignReply,
  assign,
  cardKey,
  type Failed,
  getUnnamed,
  lineClip,
  type Picked,
  profileKey,
  UNNAMED_KEY,
  type UnnamedFace,
  type UnnamedVideo,
  type UnnamedVoice,
} from '../api';
import { Empty, Section, type Status } from './parts';
import classes from './profile.module.css';
import { cameraName, failure, refusal, shortWhen } from './role';

type Key = `face:${number}` | `voice:${number}` | `video:${string}`;

/** What failed, in one sentence, for the line under the footer. */
export function failedText(failed: Failed[]): string {
  if (!failed.length) return '';
  const what = failed.map(
    f => `${f.kind === 'face' ? 'a face' : f.kind === 'voice' ? 'a voice' : 'a video'} (${f.error})`,
  );
  return `Couldn't add ${what.join(', ')}.`;
}

/** A face to pick: a strip of its pictures, and when it was seen. */
function faceTile(f: UnnamedFace, n: number, of: number): GridTile {
  const seen = [f.count === 1 ? 'Seen once' : `Seen ${f.count} times`, f.last_seen ? shortWhen(f.last_seen) : '']
    .filter(Boolean)
    .join(', ');
  return {
    key: `face:${f.id}`,
    label: `Face ${n} of ${of}: ${seen}`,
    content: (
      <span className={classes.pickBody}>
        <span className={classes.strip}>
          {f.photos.slice(0, 4).map(u => (
            <SafeImage key={u} src={u} alt="" loading="lazy" retries={2} />
          ))}
        </span>
        <span className={classes.pickMeta}>{seen}</span>
      </span>
    ),
  };
}

/** A video to pick: its poster, what happens in it, and when and where. */
function videoTile(v: UnnamedVideo): GridTile {
  const when = `${shortWhen(v.start)}, ${cameraName(v.cam, v.camera_name)}`;
  return {
    key: `video:${v.id}`,
    label: `Video: ${v.what ? `${v.what}, ` : ''}${when}, ${clipLength(v.duration_s)}`,
    content: (
      <span className={classes.pickBody}>
        <ClipPoster src={v.thumb} seconds={v.duration_s} badge={false} />
        {v.what && <span className={classes.pickWhat}>{v.what}</span>}
        <span className={classes.pickMeta}>{when}</span>
      </span>
    ),
  };
}

function VoiceRow({ v, name, on, toggle }: { v: UnnamedVoice; name: string; on: boolean; toggle: () => void }) {
  const s = v.sample;
  return (
    <li className={on ? classes.voicePicked : undefined}>
      {s ? (
        <PlayClip
          className={classes.play}
          url={lineClip(s.line)}
          playKey={`said:${s.line}`}
          ts={s.ts_start}
          label={v.label}
        />
      ) : (
        <span className={classes.play} aria-hidden="true" />
      )}
      <span className={classes.saidText}>
        <span className={classes.voiceName}>{`${v.label}, ${plural(v.lines, 'line', 'lines')}`}</span>
        {s && <span>{`“${s.text}”`}</span>}
        {v.last_heard && (
          <Text span size="xs" c="dimmed" className={classes.saidWhen}>
            {`Last heard ${shortWhen(v.last_heard)}`}
          </Text>
        )}
      </span>
      <Switch
        className={classes.thisIs}
        checked={on}
        onChange={toggle}
        label={`This is ${name}`}
        aria-label={`${v.label}: this is ${name}`}
      />
    </li>
  );
}

export function Pickers({
  name,
  onAssigned,
  onSkip,
}: {
  name: string;
  onAssigned: (reply: AssignReply) => void;
  onSkip: () => void;
}) {
  const client = useQueryClient();
  const q = useQuery({
    queryKey: UNNAMED_KEY,
    queryFn: ({ signal }) => getUnnamed({ signal }),
    retry: false,
    refetchInterval: query =>
      query.state.status !== 'error' && (query.state.data?.shots_pending || query.state.data?.videos_pending)
        ? 2000
        : false,
  });
  const loading = !q.data && (q.isPending || q.isFetching);
  const slow = useSlow(loading);
  const [picked, setPicked] = useState<ReadonlySet<Key>>(new Set());
  const [busy, setBusy] = useState(false);
  const [st, setSt] = useState<Status>(null);
  const toggle = (k: Key) =>
    setPicked(s => {
      const n = new Set(s);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });
  // A pick counts only while its item is still listed: a refresh can take one away (someone else named it).
  const listed = new Set<Key>([
    ...(q.data?.faces ?? []).filter(f => f.photos.length > 0).map(f => `face:${f.id}` as const),
    ...(q.data?.voices ?? []).map(v => `voice:${v.id}` as const),
    ...(q.data?.videos ?? []).filter(v => v.thumb).map(v => `video:${v.id}` as const),
  ]);
  const live = [...picked].filter(k => listed.has(k));
  const n = live.length;

  const add = async () => {
    const out: Picked = { faces: [], voices: [], videos: [] };
    for (const k of live) {
      const [kind, id] = [k.slice(0, k.indexOf(':')), k.slice(k.indexOf(':') + 1)];
      if (kind === 'face') out.faces.push(Number(id));
      else if (kind === 'voice') out.voices.push(Number(id));
      else out.videos.push(id);
    }
    setBusy(true);
    setSt({ text: `Adding to ${name}` });
    try {
      const j = await assign(name, out);
      if (!j.ok) {
        setSt({ err: [refusal(j, 'Adding failed.'), failedText(j.failed ?? [])].filter(Boolean).join(' ') });
        return;
      }
      setSt(null);
      void client.invalidateQueries({ queryKey: UNNAMED_KEY });
      void client.invalidateQueries({ queryKey: cardKey(name) });
      void client.invalidateQueries({ queryKey: profileKey(name) });
      onAssigned(j);
    } catch (e) {
      // "Nothing stuck" is a 400 whose body still says what failed.
      const data = e instanceof HttpError ? (e.data as { failed?: Failed[] } | null) : null;
      setSt({ err: [failure(e, 'Adding failed.'), failedText(data?.failed ?? [])].filter(Boolean).join(' ') });
    } finally {
      setBusy(false);
    }
  };

  let body: ReactNode = null;
  if (!q.data && q.isPending) body = <BlockSkeleton heading lines={6} label="Loading what nobody has named" />;
  else if (q.data) {
    const { faces, videos, voices } = q.data;
    const readyVideos = videos.filter(v => v.thumb);
    const readyFaces = faces.filter(f => f.photos.length > 0);
    body =
      !faces.length &&
      !videos.length &&
      !voices.length &&
      !q.data.incomplete &&
      !q.data.shots_pending &&
      !q.data.shots_unavailable &&
      !q.data.videos_pending &&
      !q.data.videos_unavailable ? (
        <Section title="Nothing to pick">
          <Empty>
            {`Everything seen and heard so far has a name. You can link ${name}'s face and voice later from Live or Who's this?`}
          </Empty>
        </Section>
      ) : (
        <>
          <Section title="Faces nobody's named">
            {readyFaces.length ? (
              <SelectableGrid
                label="Faces to pick"
                tiles={readyFaces.map((f, n) => faceTile(f, n + 1, readyFaces.length))}
                isOn={key => picked.has(key as Key)}
                onToggle={key => toggle(key as Key)}
                tile={170}
              />
            ) : q.data.shots_pending || q.data.shots_unavailable ? null : (
              <Empty>
                {faces.length ? 'No saved face pictures are available.' : 'Every face seen so far has a name.'}
              </Empty>
            )}
            <ProcessingImages count={q.data.shots_pending} unavailable={q.data.shots_unavailable} />
          </Section>
          <Section title="Video with nobody named">
            {readyVideos.length ? (
              <SelectableGrid
                label="Videos to pick"
                tiles={readyVideos.map(videoTile)}
                isOn={key => picked.has(key as Key)}
                onToggle={key => toggle(key as Key)}
                tile={170}
              />
            ) : q.data.videos_pending || q.data.videos_unavailable ? null : (
              <Empty>
                {q.data.incomplete ? 'Video clips could not be loaded.' : 'No video with only unnamed people in it.'}
              </Empty>
            )}
            <ProcessingImages kind="video" count={q.data.videos_pending} unavailable={q.data.videos_unavailable} />
          </Section>
          <Section title="Voices nobody's named">
            {voices.length ? (
              <ul className={classes.said}>
                {voices.map(v => (
                  <VoiceRow
                    key={v.id}
                    v={v}
                    name={name}
                    on={picked.has(`voice:${v.id}`)}
                    toggle={() => toggle(`voice:${v.id}`)}
                  />
                ))}
              </ul>
            ) : (
              <Empty>Every voice heard so far has a name.</Empty>
            )}
          </Section>
        </>
      );
  }

  return (
    <div className={classes.pickers}>
      {loading && (
        <Text role="status" aria-live="polite">
          {slow
            ? 'Still loading what nobody has named. You can keep using this page.'
            : 'Loading what nobody has named…'}
        </Text>
      )}
      {q.isError && (
        <LoadError
          message={failure(q.error, 'Could not load what nobody has named.')}
          onRetry={() => void q.refetch()}
          busy={q.isFetching}
        />
      )}
      {q.data?.incomplete && (
        <LoadError message={q.data.incomplete} onRetry={() => void q.refetch()} busy={q.isFetching} />
      )}
      {body}
      <section className={classes.footer} aria-label={`Picked for ${name}`}>
        <Text fw={500}>{`${n} picked for ${name}`}</Text>
        <Group gap="xs" ml="auto">
          <Button variant="default" onClick={onSkip} disabled={busy}>
            Skip for now
          </Button>
          <Button onClick={() => void add()} disabled={!n || busy}>
            {n ? `Add ${n} to ${name}` : `Add to ${name}`}
          </Button>
        </Group>
        <div className={classes.footerStatus} aria-live="polite">
          {st && <span className={'err' in st ? classes.err : undefined}>{'err' in st ? st.err : st.text}</span>}
        </div>
      </section>
    </div>
  );
}
