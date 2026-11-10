// Known › Activities: the owner's list of activities, which the panel adds to and deletes from. The server
// keeps it in the knowledge store (memory/activity_kinds.py), so it survives a reset; see-actions rereads it each
// minute and looks for each activity by its model classes, the ones its name matched and the ones it learned from
// the owner's answers.
import { queryOptions, type useQueryClient } from '@tanstack/react-query';
import { request } from '../../shared/http';
import type { ActivityPromotion } from '../review/api';

export interface ModelClasses {
  ntu: string[]; // NTU-120 classes, read from a skeleton
  kinetics: string[]; // Kinetics-600 classes, read from the pixels
}

export interface ActivityKind {
  kind: string; // the slug every clip, answer, and count uses
  name: string;
  description: string;
  rule: boolean; // the walk-by line: read from no model class
  classes: ModelClasses; // matched from its words when it was added
  learned: ModelClasses; // learned from the owner's answers since
  answers: number; // clips the owner answered
  promotion: ActivityPromotion | null;
}

export interface DeletedKind {
  kind: string;
  name: string;
  answers: number;
}

export interface KindsResponse {
  kinds: ActivityKind[];
  deleted: DeletedKind[];
}

export const kindsKey = ['activities', 'kinds'] as const;

export const kindsQuery = () =>
  queryOptions({
    queryKey: kindsKey,
    queryFn: ({ signal }) => request<KindsResponse>('/activity/kinds', undefined, signal),
  });

type Change = { ok: boolean; kind: ActivityKind };

export const addKind = (name: string, description: string) => request<Change>('/activity/kinds', { name, description });
export const deleteKind = (kind: string) => request<Change>('/activity/kinds/delete', { kind });
export const restoreKind = (kind: string) => request<Change>('/activity/kinds/restore', { kind });

/** After the list changes: the list, and every day's clips (a deleted kind's leave them, a restored one's return). */
export const refreshKinds = (client: ReturnType<typeof useQueryClient>) =>
  Promise.all([
    client.invalidateQueries({ queryKey: kindsKey }),
    client.invalidateQueries({
      predicate: ({ queryKey }) =>
        queryKey.length === 4 && queryKey[0] === 'review' && queryKey[1] === 'day' && queryKey[3] === 'activities',
    }),
  ]);

/** Every class an activity is looked for by, both models, defined then learned. */
export const classesOf = (k: ActivityKind): string[] => [
  ...k.classes.ntu,
  ...k.classes.kinetics,
  ...k.learned.ntu,
  ...k.learned.kinetics,
];

/** What the cameras look for, in a sentence: "Looks for squat down and squat." */
export function lookingFor(k: ActivityKind): string {
  if (k.rule) return "Counted when a person crosses a camera's counting line.";
  const all = [...new Set(classesOf(k))];
  if (!all.length) return `Nothing matches its name yet: call a clip “${k.name}” to teach it.`;
  const shown = all.length > 4 ? [...all.slice(0, 4), `${all.length - 4} more`] : all;
  const list = shown.length < 3 ? shown.join(' and ') : `${shown.slice(0, -1).join(', ')}, and ${shown.at(-1)}`;
  return `Looks for ${list}.`;
}
