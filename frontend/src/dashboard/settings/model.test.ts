import { describe, expect, it } from 'vitest';
import { type ConfigView, helpOf, initialValues, restartDestination, type Setting, sameDocument } from './model';

describe('settings drafts', () => {
  it('never initializes an editable secret from a server value', () => {
    const view: ConfigView = {
      ok: true,
      control: true,
      needs_setup: false,
      path: 'pack.toml',
      sections: [
        {
          name: 'owner',
          description: '',
          settings: [
            {
              key: 'owner.password',
              value: 'must-not-escape',
              default: '',
              type: 'str',
              secret: true,
              description: '',
            },
            { key: 'owner.name', value: 'Alice', default: '', type: 'str', secret: false, description: '' },
            { key: 'journal.levels', value: [30, 180], default: [], type: 'list', secret: false, description: '' },
          ],
        },
      ],
    };
    expect(initialValues(view)).toEqual({ 'owner.password': '', 'owner.name': 'Alice', 'journal.levels': '30, 180' });
  });
});

it.each([
  // Settings is a page of its own: it comes back where it was
  [
    'http://example.test/settings#set-recall.enabled',
    'http://example.test:9000/',
    'http://example.test:9000/settings#set-recall.enabled',
  ],
  [
    'http://example.test/review?q=day#moment',
    'http://example.test:9000/',
    'http://example.test:9000/review?q=day#moment',
  ],
  ['http://example.test/settings/', 'http://example.test:9000/', 'http://example.test:9000/settings/'],
])('follows a restart from %s to the same page on the new port', (current, reply, expected) => {
  expect(restartDestination(reply, current)).toBe(expected);
});

it('reloads after a restart that comes back to the same page, whatever its hash', () => {
  expect([
    sameDocument('http://example.test/settings#phone', 'http://example.test/settings#you'),
    sameDocument('http://example.test:9000/settings#phone', 'http://example.test/settings#phone'),
  ]).toEqual([true, false]);
});

describe("a feature's help", () => {
  const feature = (help: string, on: boolean): Setting => ({
    key: 'journal.enabled',
    type: 'bool',
    value: on,
    default: on,
    secret: false,
    description: '',
    feature: 'The journal',
    help,
  });

  it("is the feature's own words, without its default", () => {
    expect([helpOf(feature('Keep a diary', true)), helpOf(feature('', false))]).toEqual(['Keep a diary', '']);
  });
});
