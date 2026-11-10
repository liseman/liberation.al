// One card pattern for every camera in Settings' Devices: a plugged-in camera and a network camera both
// show their live preview, a Use checkbox, and a Name row in the same places.
import { Checkbox, TextInput } from '@mantine/core';
import { type ReactNode, useId, useState } from 'react';
import { LiveCam } from '../../camera/LiveCam';
import { unnamed, viewName } from '../../review/views';
import { SettingsRow } from '../layout';
import layout from '../layout.module.css';
import fields from '../settings.module.css';
import classes from './devices.module.css';

export const toggleClasses = {
  body: classes.toggleBody,
  labelWrapper: classes.toggleWords,
  label: layout.label,
  description: layout.description,
};

/** A camera card's head: its preview while used (dark while nothing comes from it), its name and status, and Use. */
export function CameraHead({
  cam,
  label,
  places,
  on,
  live,
  status,
  use,
}: {
  cam: string;
  label: string;
  places: Readonly<Record<string, string>>;
  on: boolean;
  live: boolean;
  status: { text: string; tone: 'ok' | 'warn' | 'dim' } | null;
  use: { description: string; disabled: boolean; onChange: (on: boolean) => void; note?: ReactNode };
}) {
  return (
    <SettingsRow className={`${fields.row} ${classes.camera}`}>
      <div className={classes.cameraInfo}>
        {on && (
          <div className={classes.preview}>
            {live ? (
              <LiveCam cam={cam} alt={`Preview of ${viewName(cam)}`} />
            ) : (
              <div className={classes.dark} role="img" aria-label={`No preview of ${viewName(cam)}`} />
            )}
            <span className={classes.previewLabel}>{viewName(cam, places)}</span>
          </div>
        )}
        <div className={layout.words}>
          <span className={layout.label}>{label}</span>
          {status && <span className={`${classes.status} ${classes[status.tone]}`}>{status.text}</span>}
        </div>
      </div>
      <div className={classes.toggles}>
        <Checkbox
          label="Use"
          aria-label={`Use ${label}`}
          description={use.description}
          classNames={toggleClasses}
          checked={on}
          onChange={e => use.onChange(e.currentTarget.checked)}
          disabled={use.disabled}
        />
        {use.note}
      </div>
    </SettingsRow>
  );
}

/** A camera's name field, which asks for a name while it has none. It keeps what is typed (the draft's list text is
 *  trimmed, which would eat a space just typed) and hands each change on; a name changed elsewhere (Reset) shows up
 *  again. */
export function CameraName({
  cam,
  device,
  saved,
  off,
  onChange,
}: {
  cam: string;
  device: string;
  saved: string;
  off: boolean;
  onChange: (name: string) => void;
}) {
  const id = useId();
  const [text, setText] = useState(saved);
  if (text.trim() !== saved && text.replace(/,/g, ' ').trim() !== saved) setText(saved);
  return (
    <SettingsRow className={fields.row}>
      <label htmlFor={id} className={layout.label}>
        Name ({device})
      </label>
      <TextInput
        id={id}
        placeholder={cam}
        // With no name, every page calls the camera by a generic id like cam0. It's one angle on the rig, so
        // the name is for the view, not a room.
        description={
          unnamed([cam], { [cam]: saved }).length
            ? `No name yet, so pages call it ${cam}. Name the view, like front or desk.`
            : undefined
        }
        value={text}
        onChange={e => {
          setText(e.currentTarget.value);
          onChange(e.currentTarget.value);
        }}
        disabled={off}
      />
    </SettingsRow>
  );
}
