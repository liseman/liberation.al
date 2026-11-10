// Says when Ollama is down or lacks a configured model, above the logs its callers would otherwise fill.
import { Alert } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { getStats, STATS_KEY } from '../sources/api';

export const OLLAMA_MS = 10_000;

export function OllamaAlert() {
  const { data } = useQuery({ queryKey: STATS_KEY, queryFn: getStats, refetchInterval: OLLAMA_MS });
  const o = data?.ollama;
  if (!o?.message) return null;
  const down = o.up === false;
  return (
    <Alert color={down ? 'red' : 'yellow'} title={down ? 'Ollama is down' : 'Ollama is missing a model'}>
      {o.message}
    </Alert>
  );
}
