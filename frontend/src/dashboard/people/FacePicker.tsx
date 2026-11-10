import { Button, Group, Modal, Stack, Text } from '@mantine/core';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { messageOf } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { hm } from '../lib/format';
import { refreshNames } from '../livetab/api';
import { knownPeopleQuery } from '../livetab/people';
import { viewName } from '../review/views';
import { navigate } from '../router';
import { dropWaiting } from '../whosthis/api';
import { FaceNameInput } from './FaceNameInput';
import { claimLiveFace, handoffLiveFace, type LiveFaceDraft } from './liveFace';

/** Naming is explicit, and every action retains the face selected when this dialog opened. */
export function FacePicker({ draft, onClose }: { draft: LiveFaceDraft; onClose: () => void }) {
  const client = useQueryClient();
  const people = useQuery(knownPeopleQuery());
  const [person, setPerson] = useState<string | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const assign = async (name = person) => {
    if (!name || busy) return;
    setBusy(true);
    setError('');
    try {
      await claimLiveFace(draft, name);
      dropWaiting(client, 'face', draft.source.group);
      refreshNames(client);
      void client.invalidateQueries({ queryKey: ['review', 'boxes'] });
      onClose();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal opened onClose={onClose} title="Who's this?" size="sm">
      <Stack gap="sm">
        <SafeImage
          src={draft.image}
          alt="Selected face"
          style={{ maxWidth: '100%', maxHeight: 220, objectFit: 'contain', alignSelf: 'center' }}
        />
        <Text size="xs" c="dimmed">
          {viewName(draft.source.camera)} · {hm(draft.source.at)}
        </Text>
        <FaceNameInput
          people={people.data ?? []}
          onSelect={setPerson}
          onAdd={name => void assign(name)}
          pending={people.isPending}
          disabled={busy || people.isError}
        />
        {people.isError && (
          <Text role="alert" c="red">
            {messageOf(people.error)}
          </Text>
        )}
        {error && (
          <Text role="alert" c="red">
            {error}
          </Text>
        )}
        <Group justify="space-between">
          <Button
            component="a"
            href="/people#new-person"
            variant="default"
            disabled={busy}
            onClick={event => {
              event.preventDefault();
              if (busy) return;
              handoffLiveFace(draft);
              onClose();
              navigate('/people#new-person');
            }}
          >
            More details in People
          </Button>
          <Button disabled={!person || people.isPending} loading={busy} onClick={() => void assign()}>
            Use selected person
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
