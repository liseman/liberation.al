// The live conversation: the shared transcript (transcript/Transcript.tsx) over the recent feed. Lines play when
// clicked, the list follows the newest line (or holds still when told not to), and a search dims the recent turns
// without a match and highlights the matches; older lines are Review's.
import { Anchor, Button, CloseButton, Group, Text, TextInput, Title } from '@mantine/core';
import { type ReactNode, useCallback, useMemo, useState } from 'react';
import { plural } from '../lib/format';
import { playLine, usePlayer } from '../lib/player';
import { usePrefs } from '../lib/prefs';
import { EMPTY_SPEAKERS } from '../lib/speakers';
import { useLive } from '../live/hooks';
import { href, linkClick } from '../router';
import { lineHits, lineSearch, type Turn } from '../transcript/model';
import { Transcript as Lines } from '../transcript/Transcript';
import './conversation.css';

/** meter: what the microphones hear, in the header beside the conversation. */
export function Transcript({ listening, meter = null }: { listening: boolean; meter?: ReactNode }) {
  const transcript = useLive('transcript');
  const speakers = useLive('speakers') ?? EMPTY_SPEAKERS;
  const look = usePrefs();
  const status = usePlayer(s => s.status);
  const [following, setFollowing] = useState(true);
  const [query, setQuery] = useState('');
  const searching = !!query.trim();
  const entries = transcript?.entries ?? null;
  // How many recent lines match, counted by the same search that dims and marks them in the list.
  const search = useMemo(
    () => (searching && entries ? lineSearch(entries, speakers) : null),
    [searching, entries, speakers],
  );
  const matching = useMemo(() => lineHits(search, query).size, [search, query]);
  const showStatus = look.statusLine !== false;
  const select = useCallback(
    (turns: Turn[]) => turns.filter(t => showStatus || t.id !== 'status').map(t => ({ t, other: false })),
    [showStatus],
  );
  const refresh = useCallback(() => {}, []); // corrections already read the live feed again

  return (
    <section className="transcript" aria-label="Live conversation">
      <header className="transcript-head">
        <div className="conv-title">
          <Title order={2}>Conversation</Title>
          {meter}
        </div>
        <Button
          variant="default"
          size="compact-sm"
          onClick={() => setFollowing(f => !f)}
          aria-pressed={following && !searching}
          disabled={searching}
          title="Follow new lines automatically; recording keeps running either way"
        >
          {following ? 'Following latest' : 'Follow latest'}
        </Button>
      </header>
      <Group className="transcript-search" gap="sm" wrap="wrap">
        <TextInput
          type="search"
          aria-label="Search recent conversation"
          placeholder="Find words or a speaker…"
          value={query}
          onChange={e => setQuery(e.currentTarget.value)}
          onKeyDown={e => e.key === 'Escape' && setQuery('')}
          style={{ flex: 1, minWidth: 120 }}
          // How many match, and Clear, inside the box: a line or a button appearing beside it moved the list.
          rightSectionWidth={searching ? 'auto' : undefined}
          rightSection={
            searching && (
              <Group gap={2} wrap="nowrap" pr={4}>
                <Text size="xs" c="dimmed" role="status" style={{ whiteSpace: 'nowrap' }}>
                  {matching ? plural(matching, 'match', 'matches') : 'No matches'}
                </Text>
                <CloseButton size="sm" aria-label="Clear search" onClick={() => setQuery('')} />
              </Group>
            )
          }
        />
        <Anchor size="sm" href={href('/review')} onClick={linkClick}>
          All recordings →
        </Anchor>
      </Group>
      <Lines
        entries={entries}
        speakers={speakers}
        view={String(transcript?.resets ?? 0)} // a search dims turns in place: the list holds where it is
        label="Recent conversation"
        className="feed"
        look={{
          balloons: look.transcript === 'balloons',
          compact: look.density === 'compact',
          timestamps: look.timestamps === 'always',
        }}
        select={select}
        query={query}
        empty={
          listening
            ? 'Listening. Conversation will appear here.'
            : 'No recent conversation. The microphone is not listening.'
        }
        onLine={e => playLine(e)}
        scope="live"
        onChanged={refresh}
        follow={following && !searching}
        onAtEnd={setFollowing}
      />
      {status && (
        <Text className="play-status" size="sm" c="dimmed" role="status">
          {status}
        </Text>
      )}
    </section>
  );
}
