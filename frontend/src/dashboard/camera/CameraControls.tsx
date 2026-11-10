// A camera's controls button on Live. The panel itself is lazy (CameraControlsPanel.tsx), loaded on first open.
import { ActionIcon, Popover } from '@mantine/core';
import { IconAdjustments } from '@tabler/icons-react';
import { lazy, useState } from 'react';
import { Deferred } from '../Deferred';

export type { CameraControl, CameraControlsView } from './CameraControlsPanel';

const CameraControlsPanel = lazy(() =>
  import('./CameraControlsPanel').then(module => ({ default: module.CameraControlsPanel })),
);

export function CameraControls({ camera, name }: { camera: string; name: string }) {
  const [opened, setOpened] = useState(false);
  return (
    <Popover
      opened={opened}
      onChange={setOpened}
      width="min(300px, calc(100vw - 16px))"
      position="bottom-start"
      middlewares={{ flip: true, shift: { crossAxis: true, padding: 8 } }}
      withinPortal
      trapFocus
    >
      <Popover.Target>
        <ActionIcon
          className="camera-settings"
          size="sm"
          variant="transparent"
          color="white"
          aria-label={`Controls for ${name}`}
          onClick={() => setOpened(value => !value)}
        >
          <IconAdjustments size={17} />
        </ActionIcon>
      </Popover.Target>
      <Popover.Dropdown style={{ maxHeight: 'min(70dvh, 600px, calc(100dvh - 16px))', overflowY: 'auto' }}>
        {opened && (
          <Deferred name="camera controls">
            <CameraControlsPanel key={camera} camera={camera} />
          </Deferred>
        )}
      </Popover.Dropdown>
    </Popover>
  );
}
