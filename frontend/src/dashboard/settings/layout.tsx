// Shared Settings chrome, based on Services: headings inside cards and full-width, evenly padded rows.
import type { HTMLAttributes, ReactNode } from 'react';
import { useId } from 'react';
import classes from './layout.module.css';

export interface SettingsSectionProps extends HTMLAttributes<HTMLElement> {
  title?: string;
  description?: ReactNode;
  trailing?: ReactNode;
  headingId?: string;
}

export function SettingsSection({
  title,
  description,
  trailing,
  headingId,
  className,
  children,
  ...props
}: SettingsSectionProps) {
  const generated = useId();
  const id = headingId ?? generated;
  return (
    <section
      aria-labelledby={title ? id : undefined}
      {...props}
      className={`${classes.card}${className ? ` ${className}` : ''}`}
    >
      {(title || description || trailing) && (
        <div className={classes.cardHead}>
          <div className={classes.words}>
            {title && (
              <h3 id={id} className={classes.cardTitle}>
                {title}
              </h3>
            )}
            {description && <div className={classes.description}>{description}</div>}
          </div>
          {trailing && <div className={classes.trailing}>{trailing}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export interface SettingsRowProps extends HTMLAttributes<HTMLElement> {
  component?: 'div' | 'fieldset' | 'li';
}

export function SettingsRow({ component: Component = 'div', className, children, ...props }: SettingsRowProps) {
  return (
    <Component {...props} className={`${classes.row}${className ? ` ${className}` : ''}`}>
      {children}
    </Component>
  );
}

export function SettingsHeader({ title, description, id }: { title: string; description?: ReactNode; id?: string }) {
  return (
    <header className={classes.sectionHead}>
      <h2 id={id} className={classes.sectionTitle}>
        {title}
      </h2>
      {description && <div className={classes.description}>{description}</div>}
    </header>
  );
}

export interface SourceItemProps {
  /** The source's head row: its name, and the checkbox or switch that turns it on. Always shown. */
  head: ReactNode;
  /** Whether the source is on (used, listened with). Off, only the head shows. */
  on: boolean;
  /** Its settings, shown only while it is on. */
  children?: ReactNode;
  component?: 'div' | 'li';
  'aria-label'?: string;
}

/** One source of input (a camera, a microphone, a network camera, a photo library): its name and on switch, and
 *  under them its settings while it's on. Turning a source off hides its settings rather than greying them out;
 *  they live on the server (or in the draft), so turning it back on shows them again as they were. Every source in
 *  Settings goes through this, so they all behave the same. */
export function SourceItem({ head, on, children, component: Component = 'div', ...props }: SourceItemProps) {
  return (
    <Component {...props} className={classes.source} data-on={on || undefined}>
      {head}
      {on && children}
    </Component>
  );
}
