// Stands in for `next/link`, which Pascal's editor imports: a plain link that opens outside the editor's frame.
import type { AnchorHTMLAttributes } from 'react';

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & {
  href: string | { pathname?: string };
  prefetch?: boolean;
};

export default function Link({ href, prefetch, children, ...rest }: Props) {
  void prefetch;
  return (
    <a href={typeof href === 'string' ? href : (href.pathname ?? '#')} target="_blank" rel="noreferrer" {...rest}>
      {children}
    </a>
  );
}
