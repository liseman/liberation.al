// One page frame and one look each for loading, empty, and error (#2450). Every page starts at the same left edge
// with the same header row; a load shows its skeleton under the real heading and says one quiet thing only when it
// runs long; an empty page says what will appear and how; an error says what didn't load in plain words, on the
// theme's tinted ground, with the raw reason kept behind "Show details".
import { Button, Menu } from '@mantine/core';
import { IconCheck, IconChevronDown } from '@tabler/icons-react';
import type { UseQueryResult } from '@tanstack/react-query';
import { Fragment, type ReactNode } from 'react';
import { BlockSkeleton, RowsSkeleton, useSlow } from './loading';
import classes from './page.module.css';

/** How long a page load runs before it says it is still going. */
export const PAGE_SLOW_MS = 10_000;

/** A phone's width, the app header's own breakpoint (app.module.css): a page folds its controls under it. */
export const PHONE_WIDTH = '(max-width: 720px)';

/** A page: its heading row (title, a count or date, one line of description, and its main action), then content. */
export function PageFrame({
  title,
  titleId,
  meta,
  lede,
  actions,
  width = 'default',
  className,
  children,
}: {
  /** Left out when the page draws its own PageHeader inside. */
  title?: ReactNode;
  titleId?: string;
  /** A count or a date beside the title. */
  meta?: ReactNode;
  /** One line saying what the page is for. */
  lede?: ReactNode;
  /** The page's main action, on the right. */
  actions?: ReactNode;
  /** 'wide' pages (Review, Live) run the full width; 'narrow' ones are a reading column. */
  width?: 'default' | 'narrow' | 'wide';
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={className ? `${classes.frame} ${className}` : classes.frame} data-width={width}>
      {title != null && <PageHeader title={title} titleId={titleId} meta={meta} lede={lede} actions={actions} />}
      {children}
    </div>
  );
}

/** The header row alone, for a page whose layout can't be the frame's single column. */
export function PageHeader({
  title,
  titleId,
  meta,
  lede,
  actions,
}: {
  title: ReactNode;
  titleId?: string;
  meta?: ReactNode;
  lede?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className={classes.head}>
      <div className={classes.headText}>
        <div className={classes.titleRow}>
          <h1 className={classes.title} id={titleId}>
            {title}
          </h1>
          {meta != null && meta !== false && <span className={classes.meta}>{meta}</span>}
        </div>
        {lede && <p className={classes.lede}>{lede}</p>}
      </div>
      {actions && <div className={classes.actions}>{actions}</div>}
    </header>
  );
}

/** A load in progress: a skeleton shaped like what is coming, and one line once it has run for `slowMs`. */
export function PageLoading({
  label,
  shape = 'block',
  lines,
  slowMs = PAGE_SLOW_MS,
}: {
  /** What is loading, in words: "people", "the call log". */
  label: string;
  shape?: 'block' | 'rows';
  lines?: number;
  slowMs?: number;
}) {
  const slow = useSlow(true, slowMs);
  const name = `Loading ${label}`;
  return (
    <div>
      {shape === 'rows' ? <RowsSkeleton label={name} rows={lines} /> : <BlockSkeleton label={name} lines={lines} />}
      <p className={classes.slow} role="status" aria-live="polite">
        {slow ? `Still loading ${label}. This can take a minute after a restart.` : ''}
      </p>
    </div>
  );
}

/** Nothing here yet: what will appear, and the one action that makes it appear. */
export function PageEmpty({
  title,
  children,
  action,
  compact = false,
}: {
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div className={classes.empty} data-compact={compact || undefined}>
      <p className={classes.emptyTitle}>{title}</p>
      {children && <p className={classes.emptyText}>{children}</p>}
      {action && <div className={classes.emptyAction}>{action}</div>}
    </div>
  );
}

/** Something didn't load or didn't save: what, in plain words, a retry, and the raw reason behind "Show details". */
export function PageError({
  title,
  children,
  detail,
  onRetry,
  retryLabel = 'Retry',
  busy = false,
}: {
  title: ReactNode;
  /** What it means and what to do. Defaults to saying nothing was lost; null for none. */
  children?: ReactNode;
  /** The raw reason (a server message or code), shown only on request. */
  detail?: string | null;
  onRetry?: () => void;
  retryLabel?: string;
  busy?: boolean;
}) {
  const body = children === undefined ? 'Nothing was lost. Try again in a moment.' : children;
  return (
    <div className={classes.error} role="alert">
      <div className={classes.errorRow}>
        <div>
          <p className={classes.errorTitle}>{title}</p>
          {body && <p className={classes.errorText}>{body}</p>}
        </div>
        {onRetry && (
          <Button size="compact-sm" variant="default" onClick={onRetry} loading={busy} style={{ flexShrink: 0 }}>
            {retryLabel}
          </Button>
        )}
      </div>
      {detail && (
        <details className={classes.details}>
          <summary>Show details</summary>
          <code>{detail}</code>
        </details>
      )}
    </div>
  );
}

/** A part of a page that failed to load: one sentence and a retry, in the same look as PageError. */
export function LoadError({
  message,
  hint = null,
  detail,
  onRetry,
  busy = false,
}: {
  message: string;
  /** A second line: what it means or what to do. */
  hint?: string | null;
  detail?: string | null;
  onRetry: () => void;
  busy?: boolean;
}) {
  return (
    <PageError title={message} detail={detail} onRetry={onRetry} busy={busy}>
      {hint}
    </PageError>
  );
}

/** One of a ViewMenu's choices: a label and the options to pick one of. */
export interface ViewChoice {
  label: string;
  value: string;
  options: readonly { value: string; label: string }[];
  onChange: (value: string) => void;
}

/** A page's view controls (sort, picture size, list or grouped) folded behind one View button, for a phone, where
 *  they would otherwise take a line each above the content. Each choice is a labelled run of items, the chosen one
 *  checked (and named "<label>, chosen", since Mantine's items are plain menu items); picking one leaves the menu
 *  open, so several can be changed in a row. */
export function ViewMenu({ choices, label = 'View' }: { choices: readonly ViewChoice[]; label?: string }) {
  return (
    <Menu position="bottom-end" closeOnItemClick={false} withinPortal>
      <Menu.Target>
        <Button variant="default" size="xs" rightSection={<IconChevronDown size={14} />}>
          {label}
        </Button>
      </Menu.Target>
      <Menu.Dropdown>
        {choices.map((c, i) => (
          <Fragment key={c.label}>
            {i > 0 && <Menu.Divider />}
            <Menu.Label>{c.label}</Menu.Label>
            {c.options.map(o => (
              <Menu.Item
                key={o.value}
                aria-label={o.value === c.value ? `${o.label}, chosen` : o.label}
                leftSection={<IconCheck size={14} style={{ visibility: o.value === c.value ? 'visible' : 'hidden' }} />}
                onClick={() => c.onChange(o.value)}
              >
                {o.label}
              </Menu.Item>
            ))}
          </Fragment>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}

/** A query's three states in the page's one look: its skeleton while it loads, PageError (with Retry) if it fails,
 * and its content once it has data. `label` names what loads ("the report", "people"). */
export function QueryState<T>({
  query,
  label,
  errorTitle = `Couldn't load ${label}`,
  shape,
  lines,
  loadingProps,
  children,
}: {
  query: UseQueryResult<T>;
  label: string;
  errorTitle?: ReactNode;
  shape?: 'block' | 'rows';
  lines?: number;
  /** Attributes for the wrapper around the loading skeleton, for a parent that sizes itself while it loads. */
  loadingProps?: Record<`data-${string}`, string>;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending)
    return (
      <div {...loadingProps}>
        <PageLoading label={label} shape={shape} lines={lines} />
      </div>
    );
  if (query.isError)
    return (
      <PageError
        title={errorTitle}
        detail={query.error.message}
        onRetry={() => void query.refetch()}
        busy={query.isFetching}
      />
    );
  return <>{children(query.data)}</>;
}
