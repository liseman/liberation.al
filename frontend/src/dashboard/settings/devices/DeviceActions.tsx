// What a Known › Devices card does (#3274): its Link button, the picker it opens (whose a device is, what thing it is,
// or where it lives), and the chips that show its link and location, each with a quick remove. The Link button is
// laid out like the Who's this? cards' answers (#3266): an icon over its caption and key, with a popover saying what
// it does on hover and on keyboard focus; L on a focused card opens the picker. A location is a place, or a room of
// its plan (memory/places.py): a device that never moves (a TV, a plug, a speaker) then tells where the Pack is.
// Whose it is comes from the people Liberational knows (the shared person name box, lib/PersonNameInput.tsx): the
// link stores the name as typed and never makes a person, so a name nobody has would link the device to no one.
import {
  Button,
  CloseButton,
  Group,
  Kbd,
  NativeSelect,
  Popover,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { IconLink, IconMapPin, IconTag, IconUser } from '@tabler/icons-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FormEvent, type KeyboardEvent, type MouseEvent, type ReactNode, useId, useState } from 'react';
import { messageOf } from '../../../shared/http';
import { LazyPersonNameInput } from '../../lib/LazyPersonNameInput';
import { knownPeopleQuery } from '../../livetab/people';
import cards from './deviceCards.module.css';
import {
  DEVICE_LINKS_KEY,
  type DeviceLocation,
  deviceName,
  locationWords,
  type PlaceOption,
  removeDeviceLink,
  removeDeviceLocation,
  type SeenDevice,
  saveDeviceLink,
  saveDeviceLocation,
  selectorLabel,
  type Target,
} from './links';

/** The key that opens a focused card's picker. */
export const LINK_KEY = 'L';
const LINK_HINT = 'say whose it is, name it as a thing, or say where it lives';

type LinkTarget = Target | 'place';

function useRefreshLinks() {
  const client = useQueryClient();
  return () => client.invalidateQueries({ queryKey: DEVICE_LINKS_KEY });
}

/** A place's option in "Where it lives": the whole place, then each room of its plan. The value is the place's id and
 *  the room, a line apart. */
function whereOptions(places: readonly PlaceOption[]) {
  return places.flatMap(p => [
    { value: `${p.id}\n`, label: `${p.name} (anywhere)` },
    ...p.rooms.map(room => ({ value: `${p.id}\n${room}`, label: `${room}, ${p.name}` })),
  ]);
}

/** The link form for one device: a person, a thing, or a place, then the name or where it lives, and what to
 *  recognize it by. `first` is the choice it opens on. */
export function LinkForm({
  device,
  label = deviceName(device),
  places = [],
  first = 'person',
  onDone,
}: {
  device: SeenDevice;
  label?: string;
  places?: readonly PlaceOption[];
  first?: LinkTarget;
  onDone: () => void;
}) {
  const refresh = useRefreshLinks();
  const id = useId();
  const [target, setTarget] = useState<LinkTarget>(first);
  const [name, setName] = useState('');
  const people = useQuery({ ...knownPeopleQuery(), enabled: target === 'person' });
  const known = people.data ?? [];
  // A person is one Liberational knows, picked from the list or typed in full (any case): the name sent is theirs.
  const person = known.find(k => k.toLocaleLowerCase() === name.trim().toLocaleLowerCase()) ?? null;
  const options = whereOptions(places);
  const [where, setWhere] = useState(options[0]?.value ?? '');
  const [by, setBy] = useState(0);
  const save = useMutation({
    mutationFn: async (): Promise<unknown> => {
      const selector = device.selectors[by];
      if (!selector) throw new Error('this device has no address to link it by');
      if (target === 'person') {
        if (!person) throw new Error('pick someone from the people list');
        return saveDeviceLink(selector, target, person);
      }
      if (target === 'thing') return saveDeviceLink(selector, target, name);
      const [place = '', room = ''] = where.split('\n');
      return saveDeviceLocation(selector, place, room);
    },
    onSuccess: async () => {
      await refresh();
      onDone();
    },
  });
  const ready = target === 'place' ? Boolean(where) : target === 'person' ? Boolean(person) : Boolean(name.trim());
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (ready) save.mutate();
  };
  return (
    <form onSubmit={submit} aria-label={`Link ${label}`}>
      <Stack gap="xs">
        <SegmentedControl
          size="xs"
          value={target}
          onChange={v => setTarget(v as LinkTarget)}
          data={[
            { value: 'person', label: 'A person' },
            { value: 'thing', label: 'A thing' },
            { value: 'place', label: 'A place' },
          ]}
          aria-label="Link it to"
        />
        {target === 'place' ? (
          options.length ? (
            <NativeSelect
              size="xs"
              label="Where it lives"
              value={where}
              onChange={e => setWhere(e.currentTarget.value)}
              data={options}
            />
          ) : (
            <Text size="xs" c="dimmed">
              No places yet. Walk the house with the Pack, and its rooms show here (Known › Places).
            </Text>
          )
        ) : target === 'person' ? (
          <>
            <LazyPersonNameInput
              known={known}
              value={name}
              onChange={setName}
              onPick={picked => setName(picked)}
              allowNew={false}
              label="Whose is it?"
              placeholder="Pick someone"
              emptyText={
                people.isPending ? 'Loading people…' : known.length ? 'Nobody by that name' : 'Nobody known yet'
              }
              modalFocus
            />
            {people.isSuccess && !known.length && (
              <Text size="xs" c="dimmed">
                Nobody is known by name yet. Name someone in People first, then link their devices here.
              </Text>
            )}
            {people.isError && (
              <Text size="xs" c="red" role="alert">
                Couldn't load the people list: {messageOf(people.error)}
              </Text>
            )}
          </>
        ) : (
          <TextInput
            id={`${id}-name`}
            size="xs"
            label="What is it?"
            placeholder="Device name"
            value={name}
            maxLength={80}
            onChange={e => setName(e.currentTarget.value)}
            data-autofocus
          />
        )}
        {device.selectors.length > 1 && (
          <NativeSelect
            size="xs"
            label="Recognize it by"
            value={String(by)}
            onChange={e => setBy(Number(e.currentTarget.value))}
            data={device.selectors.map((s, i) => ({ value: String(i), label: selectorLabel(s.kind) }))}
          />
        )}
        {save.error && (
          <Text size="sm" c="red" role="alert">
            Not linked: {messageOf(save.error)}
          </Text>
        )}
        <Group gap="xs">
          <Button size="xs" variant="default" type="submit" loading={save.isPending} disabled={!ready}>
            Link
          </Button>
          <Button size="xs" variant="default" onClick={onDone}>
            Cancel
          </Button>
        </Group>
      </Stack>
    </form>
  );
}

/** A card's Link button: its icon over its caption and key, with a popover saying what it does. */
export function LinkButton({ name, onClick, disabled }: { name: string; onClick: () => void; disabled?: boolean }) {
  return (
    <Tooltip
      label={`Link (${LINK_KEY}): ${LINK_HINT}`}
      multiline
      w={240}
      withArrow
      openDelay={150}
      events={{ hover: true, focus: true, touch: false }}
    >
      <UnstyledButton
        className={cards.action}
        aria-label={name}
        aria-keyshortcuts={LINK_KEY}
        onClick={onClick}
        disabled={disabled}
      >
        <IconLink size={16} aria-hidden />
        <span className={cards.caption}>
          Link
          <Kbd className={cards.kbd}>{LINK_KEY}</Kbd>
        </span>
      </UnstyledButton>
    </Tooltip>
  );
}

/** The Link button with its picker, a popover under it. `name` is the button's accessible name ("Link Pixel", or a
 *  Wi-Fi card's "Whose is Pixel?"). */
export function LinkPicker({
  device,
  label,
  name = `Link ${label}`,
  places,
  opened,
  setOpened,
}: {
  device: SeenDevice;
  label: string;
  name?: string;
  places: readonly PlaceOption[];
  opened: boolean;
  setOpened: (open: boolean) => void;
}) {
  // A device with a person or thing already opens on its place, and one with a place on whose it is.
  const first: LinkTarget = device.linked ? 'place' : 'person';
  return (
    <Popover opened={opened} onChange={setOpened} position="bottom-end" withArrow shadow="md" trapFocus width={280}>
      <Popover.Target>
        <span>
          <LinkButton name={name} onClick={() => setOpened(!opened)} disabled={device.selectors.length === 0} />
        </span>
      </Popover.Target>
      <Popover.Dropdown>
        {opened && (
          <LinkForm device={device} label={label} places={places} first={first} onDone={() => setOpened(false)} />
        )}
      </Popover.Dropdown>
    </Popover>
  );
}

/** One chip: an icon, its words, and a remove button when this machine may change it. */
function Chip({
  icon,
  text,
  title,
  remove,
  control,
}: {
  icon: ReactNode;
  text: string;
  title: string;
  remove: { label: string; run: () => Promise<unknown> };
  control: boolean;
}) {
  const refresh = useRefreshLinks();
  const removing = useMutation({ mutationFn: remove.run, onSuccess: refresh });
  return (
    <>
      <span className={cards.chip} title={title}>
        {icon}
        <span className={cards.chipText}>{text}</span>
        {control && (
          <CloseButton
            size="xs"
            aria-label={remove.label}
            disabled={removing.isPending}
            onClick={() => removing.mutate()}
          />
        )}
      </span>
      {removing.error && (
        <Text size="xs" c="red" role="alert">
          Not removed: {messageOf(removing.error)}
        </Text>
      )}
    </>
  );
}

const TARGET_WORDS: Record<Target, string> = { person: 'a person', thing: 'a thing' };

/** A device's link (a person or a thing) and where it lives, as chips with a quick remove each. */
export function LinkChips({
  label,
  linked,
  location,
  control,
}: {
  label: string;
  linked: { id: string; target: Target; name: string } | null;
  location?: DeviceLocation | null;
  control: boolean;
}) {
  if (!linked && !location) return null;
  return (
    <div className={cards.chips}>
      {linked && (
        <Chip
          icon={linked.target === 'person' ? <IconUser size={12} aria-hidden /> : <IconTag size={12} aria-hidden />}
          text={linked.name}
          title={`Linked to ${linked.name}, ${TARGET_WORDS[linked.target] ?? linked.target}`}
          remove={{ label: `Unlink ${label}`, run: () => removeDeviceLink(linked.id) }}
          control={control}
        />
      )}
      {location && (
        <Chip
          icon={<IconMapPin size={12} aria-hidden />}
          text={location.room ?? location.place}
          title={`Lives in ${locationWords(location)}`}
          remove={{ label: `Remove where ${label} lives`, run: () => removeDeviceLocation(location.id) }}
          control={control}
        />
      )}
    </div>
  );
}

/** A click on a card off its buttons and fields gives it focus, so its key works at once (as Who's this? cards do).
 *  A click in a portal (the picker, a tooltip) bubbles here through React too: that isn't the card's. */
export function focusCard(e: MouseEvent<HTMLElement>) {
  const target = e.target as HTMLElement;
  if (!e.currentTarget.contains(target) || target.closest('button, a, input, select, textarea, label')) return;
  e.currentTarget.focus({ preventScroll: true });
}

/** A card's keys: L opens its picker while the card itself (not a field in it) has focus. */
export function cardKeys(open: (() => void) | null) {
  return (e: KeyboardEvent<HTMLElement>) => {
    if (!open || e.altKey || e.ctrlKey || e.metaKey || e.nativeEvent.isComposing) return;
    if ((e.target as HTMLElement).closest('input, textarea, select, [contenteditable]')) return;
    if (e.key.toUpperCase() !== LINK_KEY) return;
    e.preventDefault();
    open();
  };
}
