import { describe, expect, it } from 'vitest';
import { kindOf } from './model';

describe('kindOf (#1331)', () => {
  it('puts each source where it belongs by what it is', () => {
    expect(kindOf('wyze')).toBe('camera');
    expect(['google_photos', 'icloud_photos', 'allsound', 'thinking_hat'].map(kindOf)).toEqual([
      'media',
      'media',
      'media',
      'media',
    ]);
    expect(kindOf('link')).toBe('none');
  });

  it('treats a source it has never heard of as imported media', () => {
    expect(kindOf('some_new_adapter')).toBe('media');
  });
});
