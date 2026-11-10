// Service selection edits configuration lists in first-run setup; Settings' Services switches them.
import { Checkbox, Text } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { useId } from 'react';
import { messageOf, request } from '../../shared/http';
import { PageError, PageLoading } from '../lib/page';
import { SettingsRow, SettingsSection } from './layout';
import layout from './layout.module.css';
import { SERVICES_KEY, type ServicesView } from './services/model';
import classes from './settings.module.css';

type Values = Record<string, string | boolean>;
type SelectionView = Omit<ServicesView, 'services'> & {
  services: (ServicesView['services'][number] & { default?: boolean })[];
};

const listOf = (value: string | boolean | undefined): string[] =>
  typeof value === 'string'
    ? value
        .split(',')
        .map(name => name.trim())
        .filter(Boolean)
    : [];

export function ServiceChecks({
  values,
  onChange,
  readOnly,
}: {
  values: Values;
  onChange: (patch: Values) => void;
  readOnly: boolean;
}) {
  const query = useQuery({
    queryKey: SERVICES_KEY,
    queryFn: ({ signal }) => request<SelectionView>('/services', undefined, signal),
  });
  const uid = useId();
  const enabled = new Set(listOf(values['services.enable']));
  const disabled = new Set(listOf(values['services.disable']));
  const error = query.error
    ? messageOf(query.error)
    : query.data?.ok === false
      ? query.data.error || 'Could not load services'
      : null;

  return (
    <SettingsSection
      title="Services on this machine"
      description="Which services Liberational installs on this machine. Turn them on and off in Services."
    >
      <fieldset aria-label="Services on this machine" disabled={readOnly} className={layout.fieldset}>
        {query.isPending && <PageLoading label="services" lines={4} />}
        {error && (
          <PageError
            title="Couldn't load the services"
            detail={error}
            onRetry={() => void query.refetch()}
            busy={query.isFetching}
          />
        )}
        {!error &&
          query.data?.services.map((service, index) => {
            const needed = service.name === 'whisper' && values['asr.engine'] === 'whisper';
            const checked = (service.default || enabled.has(service.name) || needed) && !disabled.has(service.name);
            const id = `${uid}-${index}`;
            return (
              <SettingsRow key={service.name} className={classes.row} data-toggle>
                <div className={layout.words}>
                  <label id={`${id}-label`} htmlFor={id} className={layout.label}>
                    {service.name}
                  </label>
                  <Text id={`${id}-help`} size="sm" c="dimmed">
                    {`${service.description}${needed ? ' (on because ASR uses Whisper)' : service.default ? '' : ' (off by default)'}`}
                  </Text>
                </div>
                <Checkbox
                  id={id}
                  aria-labelledby={`${id}-label`}
                  aria-describedby={`${id}-help`}
                  checked={checked}
                  disabled={readOnly}
                  onChange={event => {
                    const nextEnabled = new Set(enabled);
                    const nextDisabled = new Set(disabled);
                    nextEnabled.delete(service.name);
                    nextDisabled.delete(service.name);
                    if (event.currentTarget.checked && !service.default) nextEnabled.add(service.name);
                    if (!event.currentTarget.checked && service.default) nextDisabled.add(service.name);
                    onChange({
                      'services.enable': [...nextEnabled].join(', '),
                      'services.disable': [...nextDisabled].join(', '),
                    });
                  }}
                />
              </SettingsRow>
            );
          })}
      </fieldset>
    </SettingsSection>
  );
}
