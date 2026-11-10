// Large responses stay manageable: what sits behind a disclosure is only built once it is opened, and long lists
// show PAGE items with the rest behind "More".
import { Fragment, type ReactNode } from 'react';
import { Disclosure } from '../lib/Disclosure';
import classes from './ask.module.css';

export const PAGE = 20;

/** A disclosure whose content is built the first time it opens, and kept after. */
export function Lazy({
  summary,
  className,
  children,
}: {
  summary: ReactNode;
  className?: string;
  children: () => ReactNode;
}) {
  return (
    <Disclosure className={className ?? classes.details} summary={summary} lazy>
      {children}
    </Disclosure>
  );
}

/**
 * One page of items, then a "More (n remaining)" disclosure with the next page. With list set, the items are <li>s
 * and the disclosure is wrapped in one too, so it stays a valid child of the <ul> around them.
 */
export function Paged<T>({
  items,
  render,
  offset = 0,
  list = false,
}: {
  items: readonly T[];
  render: (item: T, index: number) => ReactNode;
  offset?: number;
  list?: boolean;
}) {
  const rest = items.length - offset - PAGE;
  const more = rest > 0 && (
    <Lazy summary={`More (${rest} remaining)`}>
      {() =>
        list ? (
          <ul>
            <Paged items={items} render={render} offset={offset + PAGE} list />
          </ul>
        ) : (
          <div>
            <Paged items={items} render={render} offset={offset + PAGE} />
          </div>
        )
      }
    </Lazy>
  );
  return (
    <>
      {items.slice(offset, offset + PAGE).map((item, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: items have no identity of their own; the page never reorders
        <Fragment key={offset + i}>{render(item, offset + i)}</Fragment>
      ))}
      {list && more ? <li>{more}</li> : more}
    </>
  );
}
