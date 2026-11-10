// The People tab: the list of everyone Liberational knows, a contact card for each, Who's this? (the faces and voices
// nobody has named yet, shared with Live: ../whosthis), and voice clones.
import { VisuallyHidden } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { messageOf } from '../../shared/http';
import { Toast } from '../lib/Toast';
import { liveKey } from '../live/feed';
import { knownTabOf } from '../router';
import { WHOS_THIS, WhosThis } from '../whosthis/WhosThis';
import { getPeople, PEOPLE_KEY, tookView, type ViewReply } from './api';
import { type LiveFaceDraft, takeLiveFace } from './liveFace';

import classes from './people.module.css';
import { Profile } from './profile/Profile';
import { NewPerson, Roster } from './Roster';
import { oneColumn } from './RosterColumn';

/** The person a link names: /people#card=<name>, else null. */
export function cardInHash(hash = location.hash): string | null {
  const name = new URLSearchParams(hash.replace(/^#/, '')).get('card')?.trim();
  return name || null;
}

/** Keep the address in step with the card shown, without a history entry each pick. */
function writeCard(name: string | null): void {
  if (cardInHash() === name) return;
  const hash = name ? `#card=${encodeURIComponent(name)}` : '';
  history.replaceState(history.state, '', location.pathname + location.search + hash);
}

export function PeopleTab() {
  const narrow = useMediaQuery('(max-width: 820px)') ?? false; // people.module.css's one column
  const client = useQueryClient();
  const [faceDraft, setFaceDraft] = useState<LiveFaceDraft | null>(null);
  const people = useQuery({ queryKey: PEOPLE_KEY, queryFn: getPeople, refetchInterval: 30000 });
  // A person's name, '?' for Who's this?, or null (Who's this? on wide windows). A link opens a card: /people#card=Kara.
  // On a phone, what waits for a name comes first, unless a link opened a card.
  const [sel, setSel] = useState<string | null>(() => cardInHash() ?? (oneColumn() ? '?' : null));
  // The selected roster card (single-click). Enter, I, or a double-click opens it.
  const [cursor, setCursor] = useState<string | null>(() => cardInHash() ?? (oneColumn() ? '?' : null));
  const [linked, setLinked] = useState<string | null>(() => cardInHash()); // the name the last link opened
  const [note, setNote] = useState<string | null>(null); // what the last delete removed
  const [fresh, setFresh] = useState<string | null>(null); // just added: their card opens on "pick what's theirs"
  const view = people.data;
  const list = view?.people ?? [];
  // Only when there is nothing to show: a failed refresh keeps the last list on screen.
  const failed = people.isError && view == null ? "Couldn't load people" : null;
  const failedDetail = failed ? messageOf(people.error) : null;
  const retry = () => void people.refetch();

  /** After a change: the reply's people view when it carries one (it is the server's own state after the change),
   * else a reload; and the live voices, whose names may have changed. */
  const refresh = (reply?: ViewReply | null) => {
    // Only a whole view: one without its unknown faces and voices would zero Who's this? until the next refresh.
    if (reply?.people && reply.unknown) tookView(client, reply);
    else void client.invalidateQueries({ queryKey: PEOPLE_KEY });
    void client.invalidateQueries({ queryKey: liveKey('speakers') });
  };
  const changed = (name?: string, reply?: ViewReply) => {
    refresh(reply);
    if (name !== undefined) {
      setSel(name);
      setCursor(name);
    }
  };
  const deleted = (reply: ViewReply, text: string) => {
    // Their lines are renamed by the microphone service a moment later, so drop them from the list right away and
    // reload once that has happened.
    refresh(reply.people ? { ...reply, people: reply.people.filter(p => p.name !== reply.name) } : reply);
    setSel(null);
    setCursor(null);
    setNote(text);
    setTimeout(() => void client.invalidateQueries({ queryKey: PEOPLE_KEY }), 3000);
  };
  const pick = (s: string) => {
    setNote(null);
    if (s !== fresh) setFresh(null);
    setCursor(s);
  };
  const open = (s: string) => {
    setNote(null);
    if (s !== fresh) setFresh(null);
    setCursor(s);
    setSel(s);
  };
  // Follow links in (Live's "Here now", and its Who's this? link) on any in-app hashchange: People stays mounted between tabs.
  useEffect(() => {
    const follow = () => {
      if (knownTabOf(location.pathname) !== 'people') return;
      if (location.hash === '#new-person' && knownTabOf(location.pathname) === 'people') {
        setFaceDraft(takeLiveFace());
        history.replaceState(history.state, '', location.pathname + location.search);
        return;
      }
      if (location.hash === `#${WHOS_THIS}` && knownTabOf(location.pathname) === 'people') {
        // A link to /people#whos-this: Who's this? here, and the address back to plain /people. (Live's own
        // /#whos-this is its rail's section, and isn't People's to take.)
        setNote(null);
        setLinked(null);
        setCursor('?');
        setSel('?');
        history.replaceState(history.state, '', location.pathname + location.search);
        return;
      }
      const name = cardInHash();
      if (!name) return;
      setNote(null);
      setLinked(name);
      setCursor(name);
      setSel(name);
    };
    follow();
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);
  // A linked name matches its person whatever the case; one nobody has falls back to Who's this?. Only a linked name:
  // a card picked here (or just renamed) is never sent away while the list catches up.
  const match = linked && sel === linked && view ? list.find(p => p.name.toLowerCase() === linked.toLowerCase()) : null;
  const shown = linked && sel === linked && view ? (match?.name ?? null) : sel;
  useEffect(() => {
    if (linked && shown && shown !== '?') setCursor(shown);
  }, [linked, shown]);
  useEffect(() => {
    if (knownTabOf(location.pathname) !== 'people') return;
    writeCard(shown && shown !== '?' ? shown : null);
  }, [shown]);
  const whos = shown == null || shown === '?';
  // Who's this? stays unmounted until the roster view is on screen. Its group reads would otherwise take both JSON
  // slots ahead of /people, and mounting it when the roster has failed leaves the loading placeholder up.
  // On a phone, .picked hides the roster. Selecting Who's this before that view exists would hide the loader
  // and, if the load fails, Retry. Keep the list layout until the view is here, then open Who's this.
  const openDetail = shown != null && !(shown === '?' && view == null);
  return (
    <div className={`${classes.page} ${openDetail ? classes.picked : ''}`}>
      {/* On a phone the roster, and its h1, hide while this shows: the page's heading for screen readers then */}
      {narrow && openDetail && <VisuallyHidden component="h1">People</VisuallyHidden>}
      <NewPerson
        key={faceDraft?.source.at ?? 'none'}
        opened={faceDraft != null}
        faceDraft={faceDraft}
        onClose={() => setFaceDraft(null)}
        onAdded={name => {
          setLinked(null);
          setFresh(null);
          setCursor(name);
          setSel(name);
          refresh();
        }}
      />
      <div className={classes.main}>
        <div className={classes.cols}>
          <Roster
            people={list}
            loaded={view != null}
            error={failed}
            errorDetail={failedDetail}
            retrying={people.isFetching}
            onRetry={retry}
            cursor={cursor}
            sel={shown}
            onPick={pick}
            onOpen={open}
            onAdded={name => {
              setFresh(name);
              void client.invalidateQueries({ queryKey: PEOPLE_KEY }).then(() => {
                setCursor(name);
                setSel(name);
              });
            }}
          />
          <div className={classes.detail}>
            {whos && view != null ? (
              <WhosThis wide onChanged={() => refresh()} onBack={() => setSel(null)} />
            ) : whos ? null : (
              <Profile
                key={shown}
                name={shown}
                person={list.find(p => p.name === shown)}
                people={list}
                onBack={() => setSel(null)}
                onChanged={changed}
                onDeleted={deleted}
                isNew={fresh === shown}
                onPickDone={() => setFresh(null)}
              />
            )}
          </div>
        </div>
      </div>
      <Toast text={note} onClose={() => setNote(null)} />
    </div>
  );
}
