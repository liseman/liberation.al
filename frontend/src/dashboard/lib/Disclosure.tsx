// A shared disclosure: Mantine handles the collapsed panel, while the trigger remains a normal keyboard button.
import { Collapse, UnstyledButton, useMantineEnv } from '@mantine/core';
import { IconChevronRight } from '@tabler/icons-react';
import { type ReactNode, useId, useState } from 'react';
import classes from './disclosure.module.css';

export function Disclosure({
  summary,
  children,
  className,
  buttonClassName,
  opened,
  defaultOpened = false,
  onChange,
  lazy = false,
  keepMounted = true,
  withChevron = true,
}: {
  summary: ReactNode;
  children: ReactNode | (() => ReactNode);
  className?: string;
  buttonClassName?: string;
  opened?: boolean;
  defaultOpened?: boolean;
  onChange?: (opened: boolean) => void;
  lazy?: boolean;
  keepMounted?: boolean;
  withChevron?: boolean;
}) {
  const [localOpened, setLocalOpened] = useState(defaultOpened);
  const [built, setBuilt] = useState(defaultOpened || opened === true);
  const open = opened ?? localOpened;
  const id = useId();
  const env = useMantineEnv();
  return (
    <div className={className} data-disclosure>
      <UnstyledButton
        className={buttonClassName ?? classes.trigger}
        data-disclosure-trigger
        aria-expanded={open}
        aria-controls={id}
        onClick={() => {
          if (!open) setBuilt(true);
          setLocalOpened(!open);
          onChange?.(!open);
        }}
      >
        {withChevron && <IconChevronRight size="1em" aria-hidden className={classes.chevron} data-open={open} />}
        {summary}
      </UnstyledButton>
      <Collapse
        id={id}
        expanded={open}
        keepMounted={keepMounted}
        keepMountedMode="display-none"
        transitionDuration={env === 'test' ? 0 : 200}
      >
        {(!lazy || built || open) && (typeof children === 'function' ? children() : children)}
      </Collapse>
    </div>
  );
}
