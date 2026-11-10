// A Wi-Fi access point's name where Known › Devices shows it ("via Living room"), with a rename beside it on the machine
// Liberational runs on. The name is the owner's (devices.access_points), set through POST /device-access-point by the
// access point's MAC (`place_id`); a Bluetooth place has no `place_id` (it's a sensor's name), so it shows as text.
import { ActionIcon, Button, Group, Popover, Text, TextInput } from '@mantine/core';
import { IconPencil } from '@tabler/icons-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, useState } from 'react';
import { messageOf, request } from '../../../shared/http';
import { DEVICE_LINKS_KEY } from './links';

export const nameAccessPoint = (mac: string, name: string) =>
  request<{ ok: true; access_points: Record<string, string> }>('/device-access-point', { mac, name: name.trim() });

/** An access point's shown name is its MAC until someone names it (or the router does): start the box empty then. */
const startName = (place: string, mac: string) => (place.toUpperCase() === mac.toUpperCase() ? '' : place);

export function PlaceName({ place, placeId, control }: { place: string; placeId?: string | null; control: boolean }) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const save = useMutation({
    mutationFn: (mac: string) => nameAccessPoint(mac, name),
    onSuccess: async () => {
      // Comings, whereabouts, presence, and the link listing all sit under this key: each shows the new name.
      await client.invalidateQueries({ queryKey: DEVICE_LINKS_KEY });
      setOpen(false);
    },
  });
  if (!control || !placeId) return <>{place}</>;
  const toggle = () => {
    if (!open) {
      setName(startName(place, placeId));
      save.reset();
    }
    setOpen(!open);
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    save.mutate(placeId);
  };
  return (
    <>
      {place}{' '}
      <Popover opened={open} onChange={setOpen} position="bottom-start" withArrow trapFocus returnFocus>
        <Popover.Target>
          <ActionIcon
            size="xs"
            variant="subtle"
            color="gray"
            onClick={toggle}
            aria-label={`Rename access point ${place}`}
            style={{ verticalAlign: 'middle' }}
          >
            <IconPencil size={12} />
          </ActionIcon>
        </Popover.Target>
        <Popover.Dropdown>
          <form onSubmit={submit} aria-label={`Rename access point ${place}`}>
            <TextInput
              size="xs"
              label="Access point name"
              description={`Shown for ${placeId.toUpperCase()}. Leave it empty for the router's name, or the address.`}
              placeholder="Living room"
              value={name}
              maxLength={80}
              onChange={e => setName(e.currentTarget.value)}
              data-autofocus
              w={260}
            />
            {save.error && (
              <Text size="sm" c="red" role="alert" mt={4}>
                Not renamed: {messageOf(save.error)}
              </Text>
            )}
            <Group gap="xs" mt="xs">
              <Button size="xs" type="submit" loading={save.isPending}>
                Save
              </Button>
              <Button size="xs" variant="default" onClick={() => setOpen(false)}>
                Cancel
              </Button>
            </Group>
          </form>
        </Popover.Dropdown>
      </Popover>
    </>
  );
}
