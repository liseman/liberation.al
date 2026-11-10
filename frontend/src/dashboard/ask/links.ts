// Link and URL guards for the Ask widget. Hrefs in answers and evidence come from model output, so they stay strict.

/** A same-origin link into the dashboard's own pages (a path or a #t= / #day= hash), or null. */
export function safeHref(value: unknown): string | null {
  // biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are what this rejects
  if (typeof value !== 'string' || /[\s\\\x00-\x1f]/.test(value) || value.startsWith('//')) return null;
  if (!value.startsWith('/') && !value.startsWith('#')) return null;
  try {
    const u = new URL(value, location.href);
    if (u.origin !== location.origin) return null;
    if (value.startsWith('#')) return /^#(?:t=|day=)[\w.=&%-]+$/.test(value) ? value : null;
    return /^\/(?:review(?:\.html)?|transcript|audio|speakers|people|recordings|calls\/audio|camera\/query)(?:\/|$)/.test(
      u.pathname,
    )
      ? u.pathname + u.search + u.hash
      : null;
  } catch {
    return null;
  }
}

/** The server's poll URL for a job, or null when it is not exactly /pack-ask/jobs/<id>. */
export function jobURL(value: unknown): string | null {
  return typeof value === 'string' && /^\/pack-ask\/jobs\/[A-Za-z0-9_-]+$/.test(value) ? value : null;
}

/** A link a model may have written into its answer: same-origin dashboard links, or http(s) without credentials. */
export function answerHref(value: string): string | null {
  if (!/^https?:\/\//i.test(value)) return safeHref(value);
  try {
    const u = new URL(value);
    return u.username || u.password ? null : u.href;
  } catch {
    return null;
  }
}

/** The app path a safe href opens in place (a tab of this app), or null when it is a plain link. */
export function tabPath(safe: string): string | null {
  const u = new URL(safe, location.href);
  if (/^\/review(?:\.html)?$/.test(u.pathname)) return `/review${u.search}${u.hash}`;
  if (u.pathname === '/people') return `/people${u.search}${u.hash}`;
  return null;
}
