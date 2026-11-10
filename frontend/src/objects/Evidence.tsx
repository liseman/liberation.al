import { Stack, Text } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { evidenceUrl, type Sighting } from './api';
import { sightingLabel } from './format';
import { draw, loadImage } from './image';
import classes from './objects.module.css';

/** A saved observation: its label, then the evidence image with the matched box drawn on it. */
export function Evidence({ sighting }: { sighting: Sighting }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [missing, setMissing] = useState(false);
  useEffect(() => {
    let live = true;
    setMissing(false);
    loadImage(evidenceUrl(sighting.evidence)).then(
      img => {
        const c = canvas.current;
        if (!live || !c) return;
        const [x, y, w, h] = sighting.bbox;
        const W = img.naturalWidth;
        const H = img.naturalHeight;
        draw(c, img, [x * W, y * H, w * W, h * H]);
      },
      () => live && setMissing(true),
    );
    return () => {
      live = false;
    };
  }, [sighting]);
  const label = sightingLabel(sighting);
  return (
    <Stack gap="xs" mt="md" pt="sm" style={{ borderTop: '1px solid var(--mantine-color-default-border)' }}>
      <Text size="sm">
        {label}
        {missing && ' · image unavailable'}
      </Text>
      <canvas ref={canvas} className={classes.canvas} role="img" aria-label={`Evidence: ${label}`} />
    </Stack>
  );
}
