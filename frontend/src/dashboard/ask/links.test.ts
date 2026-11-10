import { describe, expect, it } from 'vitest';
import { answerHref, jobURL, safeHref } from './links';

describe('safeHref', () => {
  it.each([
    ['/review#t=1700000000', '/review#t=1700000000'],
    ['/review.html?day=2026-10-20', '/review.html?day=2026-10-20'],
    ['/calls/audio/x.wav', '/calls/audio/x.wav'],
    ['/camera/query?cam=cam0', '/camera/query?cam=cam0'],
    ['#day=2026-10-23', '#day=2026-10-23'],
    ['#t=1700000000&day=2026-10-23', '#t=1700000000&day=2026-10-23'],
  ])('keeps the dashboard link %s', (value, expected) => {
    expect(safeHref(value)).toBe(expected);
  });

  it.each([
    'javascript:alert(1)',
    '//evil.test/',
    'https://evil.test/review',
    'http://localhost:3000/review',
    '/etc/passwd',
    '/reviewer',
    '/review x',
    '/review\\x',
    '/review\u0000',
    '#x',
    '#t=<b>',
    'review#t=1',
    '',
  ])('drops %j', value => {
    expect(safeHref(value)).toBeNull();
  });

  it('drops anything that is not a string', () => {
    expect(safeHref(null)).toBeNull();
    expect(safeHref({ toString: () => '/review' })).toBeNull();
  });
});

describe('jobURL', () => {
  it('accept only their own job paths', () => {
    expect(jobURL('/pack-ask/jobs/a1_B-2')).toBe('/pack-ask/jobs/a1_B-2');
    expect(jobURL('/pack-ask/jobs/../x')).toBeNull();
    expect(jobURL('https://evil.test/pack-ask/jobs/a')).toBeNull();
    expect(jobURL('/pack-ask/jobs/a?x=1')).toBeNull();
  });
});

describe('answerHref', () => {
  it('allows http(s) without credentials, and dashboard links', () => {
    expect(answerHref('https://example.org/a b')).toBe('https://example.org/a%20b');
    expect(answerHref('HTTP://example.org')).toBe('http://example.org/');
    expect(answerHref('https://user:pw@example.org/')).toBeNull();
    expect(answerHref('/review#t=1')).toBe('/review#t=1');
    expect(answerHref('ftp://example.org/')).toBeNull();
    expect(answerHref('mailto:a@example.org')).toBeNull();
  });
});
