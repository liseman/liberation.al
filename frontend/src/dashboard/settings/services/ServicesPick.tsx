// The controls for services.enable ("Also run") and services.disable ("Don't run"): a pick from this
// machine's services, so a name can't be mistyped. Also run offers the services that are off by default, Don't run
// the ones that are on; one another setting needs (Whisper, for ASR) is offered by neither. Saved with the draft.
import { MultiSelect, TagsInput } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import type { ControlProps } from '../controls';
import { useDraft } from '../draft';
import { listItems, listText } from '../model';
import { SECTION_POLL_MS, servicesQuery } from './api';
import { pickable } from './model';

export function ServicesPick({ row, id, labelled, disabled, error, inputClass }: ControlProps) {
  const d = useDraft();
  const query = useQuery(servicesQuery(SECTION_POLL_MS));
  const view = query.data;
  const value = [...new Set(listItems(d.value(row.key)))]; // a name saved twice is one pick
  const set = (names: string[]) => d.set(row.key, listText(names));
  const common = { id, 'aria-labelledby': labelled, disabled, error, classNames: { input: inputClass } };
  if (query.isError || view?.ok === false)
    return <TagsInput {...common} value={value} onChange={set} splitChars={[',']} />; // can't list them: type names
  const offered = pickable(view?.services ?? [], row.key === 'services.disable' ? 'on' : 'off');
  // A name saved earlier that this machine doesn't offer stays, so a pick never drops it unasked.
  const data = [
    ...offered.map(s => ({ value: s.name, label: s.name })),
    ...value.filter(n => !offered.some(s => s.name === n)).map(n => ({ value: n, label: n })),
  ];
  return (
    <MultiSelect
      {...common}
      data={data}
      value={value}
      onChange={set}
      searchable
      placeholder={value.length ? undefined : 'None'}
      nothingFoundMessage={view ? 'No such service' : 'Loading the services...'}
    />
  );
}
