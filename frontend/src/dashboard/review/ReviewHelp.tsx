// Review's "?" on the control bar (#2454): the keyboard shortcuts and the key to the timeline, out of the way until
// someone asks. The shortcuts come from the same table the engine dispatches on (reviewKeys.ts, #2285). Laid out as
// two aligned columns (#2577): the keys or the swatch on the left, what it means on the right, in short sections.
import { ActionIcon, Kbd, Popover } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconQuestionMark } from '@tabler/icons-react';
import { Fragment } from 'react';
import classes from './review.module.css';
import { KEY_GROUPS, REVIEW_KEYS } from './reviewKeys';
import { TimelineKey } from './Timeline';

/** Review's keys, a section per group, each key or pair of keys beside what it does. */
export function KeyHelp() {
  return (
    <table className={classes.helpTable} aria-label="Keyboard shortcuts">
      {KEY_GROUPS.map(group => (
        <tbody key={group.id}>
          <tr>
            <th colSpan={2} scope="rowgroup" className={classes.helpGroup}>
              {group.title}
            </th>
          </tr>
          {REVIEW_KEYS.filter(k => k.group === group.id).map(({ id, shown, does }) => (
            <tr key={id}>
              <td className={classes.helpLead}>
                {shown.map((label, n) => (
                  <Fragment key={label}>
                    {n > 0 && <span className={classes.keyOr}> or </span>}
                    <Kbd size="xs">{label}</Kbd>
                  </Fragment>
                ))}
              </td>
              <td>{does}</td>
            </tr>
          ))}
        </tbody>
      ))}
    </table>
  );
}

/** The "?" button and its popover. timeline: also explain the timeline (Review's page, not an embedded clip). */
export function ReviewHelp({ timeline = false }: { timeline?: boolean }) {
  // Beside the button on a wide screen, where the whole table fits beside the video; below it on a narrow one. Either
  // way no taller than the window, scrolling inside if it must.
  const wide = useMediaQuery('(min-width: 1000px)');
  return (
    <Popover
      position={wide ? 'left' : 'bottom-end'}
      withArrow
      shadow="md"
      width="min(360px, calc(100vw - 16px))"
      middlewares={{ flip: true, shift: { padding: 8 }, size: { padding: 8 } }}
      // A panel to read and scroll, so it takes focus when it opens and gives it back to "?" when it closes (Escape,
      // or a click outside), as a dialog does: keyboard users land in it and can scroll it (the pass, #2621).
      trapFocus
      returnFocus
    >
      <Popover.Target>
        <ActionIcon variant="default" size={28} radius="xl" aria-label="Shortcuts and timeline help">
          <IconQuestionMark size={16} aria-hidden />
        </ActionIcon>
      </Popover.Target>
      {/* The dropdown keeps the size middleware's max height; the scrolling is on a wrapper inside it, so the arrow
          (a child of the dropdown, half outside its edge) isn't scrollable overflow and adds no scrollbar. */}
      <Popover.Dropdown className={`${classes.help} ${classes.helpPanel}`}>
        {/* biome-ignore lint/a11y/noNoninteractiveTabindex: a scrolling region takes focus so the keyboard can scroll it */}
        <section className={classes.helpScroll} tabIndex={0} aria-label="Shortcuts and timeline help">
          <KeyHelp />
          {timeline && <TimelineKey />}
        </section>
      </Popover.Dropdown>
    </Popover>
  );
}
