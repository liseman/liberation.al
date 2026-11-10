import { Kbd, UnstyledButton } from '@mantine/core';
import {
  type KeyboardEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { Box } from './api';
import classes from './editor.module.css';
import {
  type CropHandle,
  clampFramePoint,
  clampImageView,
  cropFromPoints,
  fitImageView,
  frameToView,
  type ImageView,
  moveFrameCrop,
  type Point,
  panImageView,
  pinchImageView,
  resizeFrameCrop,
  type Size,
  viewToFrame,
  zoomImageView,
} from './geometry';

interface FrameEditorProps {
  image: HTMLImageElement;
  overlay?: HTMLImageElement | null;
  /** SAM selection in normalized full-frame coordinates. */
  selectionBox?: Box | null;
  /** Manual crop in full-resolution frame pixels, without a preview caption strip. */
  crop: Box | null;
  onCropChange: (crop: Box | null) => void;
  onPick: (x: number, y: number) => void;
  /** SAM is working: picking pauses, but view and crop editing stay available. */
  busy?: boolean;
  /** Enrollment is saving: all editor controls and gestures pause. */
  disabled?: boolean;
  /** A new SAM job returns to pan/pick mode without discarding the crop or view. */
  selectRequest?: number;
  /** Find objects in the region in view, as integer frame pixels [x, y, w, h]. */
  onFindObjects?: (region: Box) => void;
  /** Objects found: outlines (normalized polygons) a click inside picks, the smallest one under the pointer first. */
  found?: { id: string; outline: [number, number][][] }[];
  /** The found object picked, drawn highlighted. */
  chosen?: string | null;
  onChoose?: (id: string) => void;
}

type Gesture =
  | { kind: 'pan'; id: number; point: Point; view: ImageView; moved: boolean; suppressPick: boolean }
  | { kind: 'crop'; id: number; point: Point }
  | { kind: 'resize'; id: number; point: Point; box: Box; handle: CropHandle | 'move' }
  | { kind: 'pinch'; center: Point; distance: number; view: ImageView };

const HANDLES: { key: CropHandle; label: string; x: number; y: number }[] = [
  { key: 'nw', label: 'top left', x: 0, y: 0 },
  { key: 'n', label: 'top', x: 0.5, y: 0 },
  { key: 'ne', label: 'top right', x: 1, y: 0 },
  { key: 'e', label: 'right', x: 1, y: 0.5 },
  { key: 'se', label: 'bottom right', x: 1, y: 1 },
  { key: 's', label: 'bottom', x: 0.5, y: 1 },
  { key: 'sw', label: 'bottom left', x: 0, y: 1 },
  { key: 'w', label: 'left', x: 0, y: 0.5 },
];
/** A polygon's area (shoelace), for picking the smallest outline under the pointer. */
const area = (polygon: [number, number][]) =>
  Math.abs(
    polygon.reduce((sum, [x, y], i) => {
      const [nx, ny] = polygon[(i + 1) % polygon.length] ?? [x, y];
      return sum + x * ny - nx * y;
    }, 0),
  ) / 2;
/** Whether a normalized point is inside a polygon (even-odd ray casting). */
const inside = ([px, py]: Point, polygon: [number, number][]) => {
  let hit = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i] ?? [0, 0];
    const [xj, yj] = polygon[j] ?? [0, 0];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
};
/** The visible part of the frame, in integer frame pixels [x, y, w, h], at least 2 x 2. */
export function visibleRegion(view: ImageView, extent: Size, size: Size): Box {
  const x0 = Math.max(0, Math.floor(-view.x / view.scale));
  const y0 = Math.max(0, Math.floor(-view.y / view.scale));
  const x1 = Math.min(size.width, Math.ceil((extent.width - view.x) / view.scale));
  const y1 = Math.min(size.height, Math.ceil((extent.height - view.y) / view.scale));
  const x = Math.min(x0, size.width - 2);
  const y = Math.min(y0, size.height - 2);
  return [x, y, Math.max(2, x1 - x), Math.max(2, y1 - y)];
}
const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const midpoint = (a: Point, b: Point): Point => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

export function FrameEditor({
  image,
  overlay = null,
  selectionBox = null,
  crop,
  onCropChange,
  onPick,
  busy = false,
  disabled = false,
  selectRequest = 0,
  onFindObjects,
  found = [],
  chosen = null,
  onChoose,
}: FrameEditorProps) {
  const size: Size = useMemo(
    () => ({ width: image.naturalWidth || image.width || 1, height: image.naturalHeight || image.height || 1 }),
    [image],
  );
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLFieldSetElement>(null);
  const [extent, setExtent] = useState<Size>({ width: size.width, height: size.height });
  const [view, setView] = useState<ImageView>(() => fitImageView(size, size));
  const viewRef = useRef(view);
  const extentRef = useRef(extent);
  const pointers = useRef(new Map<number, Point>());
  const gesture = useRef<Gesture | null>(null);
  const suppressClick = useRef(false);
  const previousRequest = useRef(selectRequest);
  const [mode, setMode] = useState<'pan' | 'crop'>('pan');
  const fit = fitImageView(size, extent);
  const setCamera = (next: ImageView) => {
    viewRef.current = next;
    setView(next);
  };
  const local = (clientX: number, clientY: number): Point => {
    const rect = canvas.current?.getBoundingClientRect();
    return [clientX - (rect?.left ?? 0), clientY - (rect?.top ?? 0)];
  };
  const resetView = () => setCamera(fitImageView(size, extent));
  const clearCrop = () => {
    suppressClick.current = pointers.current.size > 0;
    gesture.current = null;
    pointers.current.clear();
    onCropChange(null);
    setMode('pan');
  };
  const pick = (point: Point) => {
    if (busy || disabled || mode !== 'pan') return;
    const [x, y] = viewToFrame(point, viewRef.current);
    if (!(x >= 0 && y >= 0 && x < size.width && y < size.height)) return;
    const at: Point = [x / size.width, y / size.height];
    const hits = found
      .filter(f => f.outline.some(polygon => inside(at, polygon)))
      .sort((a, b) => a.outline.reduce((s, p) => s + area(p), 0) - b.outline.reduce((s, p) => s + area(p), 0));
    const hit = hits[0];
    if (hit && onChoose) onChoose(hit.id);
    else onPick(x, y);
  };

  useLayoutEffect(() => {
    const measure = () => {
      const rect = canvas.current?.getBoundingClientRect();
      const parent = viewport.current?.getBoundingClientRect();
      const next = {
        width: rect?.width || parent?.width || size.width,
        height: rect?.height || parent?.height || size.height,
      };
      const oldFit = fitImageView(size, extentRef.current);
      const current = viewRef.current;
      const nextView =
        Math.abs(current.scale - oldFit.scale) < 1e-8 ? fitImageView(size, next) : clampImageView(current, size, next);
      extentRef.current = next;
      setExtent(next);
      viewRef.current = nextView;
      setView(nextView);
    };
    measure();
    const observer = new ResizeObserver(measure);
    if (viewport.current) observer.observe(viewport.current);
    return () => observer.disconnect();
  }, [size]);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext('2d');
    if (!element || !context) return;
    const dpr = window.devicePixelRatio || 1;
    element.width = Math.round(extent.width * dpr);
    element.height = Math.round(extent.height * dpr);
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, element.width, element.height);
    context.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.x, dpr * view.y);
    context.drawImage(image, 0, 0, size.width, size.height);
    if (overlay) context.drawImage(overlay, 0, 0, size.width, size.height);
    for (const f of found) {
      const picked = f.id === chosen;
      context.lineWidth = (picked ? 3 : 1.5) / view.scale;
      context.strokeStyle = picked ? '#6fd3c3' : '#ffd43b';
      for (const polygon of f.outline) {
        context.beginPath();
        polygon.forEach(([x, y], i) => {
          if (i === 0) context.moveTo(x * size.width, y * size.height);
          else context.lineTo(x * size.width, y * size.height);
        });
        context.closePath();
        context.stroke();
      }
    }
    if (selectionBox) {
      context.lineWidth = 2 / view.scale;
      context.strokeStyle = '#6fd3c3';
      context.strokeRect(
        selectionBox[0] * size.width,
        selectionBox[1] * size.height,
        selectionBox[2] * size.width,
        selectionBox[3] * size.height,
      );
    }
  }, [image, overlay, selectionBox, view, extent, size, found, chosen]);

  const wheel = useCallback(
    (event: WheelEvent) => {
      if (disabled) return;
      event.preventDefault();
      const rect = canvas.current?.getBoundingClientRect();
      const anchor: Point = [event.clientX - (rect?.left ?? 0), event.clientY - (rect?.top ?? 0)];
      const next = zoomImageView(
        viewRef.current,
        anchor,
        viewRef.current.scale * Math.exp(-event.deltaY * 0.002),
        size,
        extent,
      );
      viewRef.current = next;
      setView(next);
    },
    [disabled, size, extent],
  );
  useEffect(() => {
    const element = viewport.current;
    element?.addEventListener('wheel', wheel, { passive: false });
    return () => element?.removeEventListener('wheel', wheel);
  }, [wheel]);
  useEffect(() => {
    if (disabled) {
      suppressClick.current = pointers.current.size > 0;
      gesture.current = null;
      pointers.current.clear();
    }
  }, [disabled]);

  useEffect(() => {
    if (previousRequest.current === selectRequest) return;
    previousRequest.current = selectRequest;
    gesture.current = null;
    pointers.current.clear();
    suppressClick.current = false;
    setMode('pan');
  }, [selectRequest]);

  function pointerDown(event: PointerEvent<HTMLElement>) {
    if (disabled || (event.pointerType !== 'touch' && event.button !== 0)) return;
    if ((event.target as HTMLElement).closest('[data-editor-control]')) return;
    event.preventDefault();
    const handleElement = (event.target as HTMLElement).closest<HTMLElement>('[data-crop-handle]');
    (handleElement ?? viewport.current)?.focus({ preventScroll: true });
    viewport.current?.setPointerCapture?.(event.pointerId);
    const point = local(event.clientX, event.clientY);
    pointers.current.set(event.pointerId, point);
    if (pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      if (a && b)
        gesture.current = {
          kind: 'pinch',
          center: midpoint(a, b),
          distance: Math.max(1, distance(a, b)),
          view: viewRef.current,
        };
      suppressClick.current = true;
      return;
    }
    const handle = handleElement?.dataset.cropHandle as CropHandle | 'move' | undefined;
    const frame = clampFramePoint(viewToFrame(point, viewRef.current), size);
    if (mode === 'crop' && crop && handle)
      gesture.current = { kind: 'resize', id: event.pointerId, point: frame, box: crop, handle };
    else if (mode === 'crop') gesture.current = { kind: 'crop', id: event.pointerId, point: frame };
    else
      gesture.current = {
        kind: 'pan',
        id: event.pointerId,
        point,
        view: viewRef.current,
        moved: false,
        suppressPick: false,
      };
    suppressClick.current = false;
  }
  function pointerMove(event: PointerEvent<HTMLElement>) {
    if (disabled || !pointers.current.has(event.pointerId)) return;
    const point = local(event.clientX, event.clientY);
    pointers.current.set(event.pointerId, point);
    const active = gesture.current;
    if (!active) return;
    if (active.kind === 'pinch') {
      const [a, b] = [...pointers.current.values()];
      if (a && b)
        setCamera(
          pinchImageView(active.view, active.center, midpoint(a, b), distance(a, b) / active.distance, size, extent),
        );
    } else if (active.kind === 'pan') {
      const delta: Point = [point[0] - active.point[0], point[1] - active.point[1]];
      if (Math.hypot(...delta) > 4) active.moved = true;
      if (active.moved) {
        suppressClick.current = true;
        setCamera(panImageView(active.view, delta, size, extent));
      }
    } else {
      const frame = clampFramePoint(viewToFrame(point, viewRef.current), size);
      if (active.kind === 'crop') onCropChange(cropFromPoints(active.point, frame, size));
      else {
        const delta: Point = [frame[0] - active.point[0], frame[1] - active.point[1]];
        onCropChange(
          active.handle === 'move'
            ? moveFrameCrop(active.box, delta, size)
            : resizeFrameCrop(active.box, active.handle, delta, size),
        );
      }
      suppressClick.current = true;
    }
  }
  function pointerUp(event: PointerEvent<HTMLElement>) {
    if (!pointers.current.has(event.pointerId)) return;
    const active = gesture.current;
    pointers.current.delete(event.pointerId);
    if (active?.kind === 'pan' && !active.moved && !active.suppressPick && event.type !== 'pointercancel') {
      pick(local(event.clientX, event.clientY));
    }
    suppressClick.current = true;
    const remaining = [...pointers.current.entries()][0];
    gesture.current = remaining
      ? { kind: 'pan', id: remaining[0], point: remaining[1], view: viewRef.current, moved: false, suppressPick: true }
      : null;
  }
  function keyDown(event: KeyboardEvent<HTMLElement>) {
    if (disabled) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      clearCrop();
      return;
    }
    if (event.target !== viewport.current && event.target !== canvas.current) return;
    const delta = (
      { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] } as Record<string, Point>
    )[event.key];
    if (delta) {
      event.preventDefault();
      setCamera(panImageView(viewRef.current, delta, size, extent));
    } else if (['+', '=', '-', '0', 'Home'].includes(event.key)) {
      event.preventDefault();
      if (event.key === '0' || event.key === 'Home') resetView();
      else
        setCamera(
          zoomImageView(
            viewRef.current,
            [extent.width / 2, extent.height / 2],
            viewRef.current.scale * (event.key === '-' ? 0.8 : 1.25),
            size,
            extent,
          ),
        );
    }
  }
  const cropOrigin = crop ? frameToView([crop[0], crop[1]], view) : null;
  return (
    <fieldset className={classes.editor} aria-label="Image editing controls" onKeyDown={keyDown}>
      <div className={classes.toolbar} role="toolbar" aria-label="Frame controls">
        <UnstyledButton
          type="button"
          disabled={disabled}
          onClick={() =>
            setCamera(
              zoomImageView(
                viewRef.current,
                [extent.width / 2, extent.height / 2],
                viewRef.current.scale * 1.25,
                size,
                extent,
              ),
            )
          }
        >
          Zoom in
        </UnstyledButton>
        <UnstyledButton
          type="button"
          disabled={disabled}
          onClick={() =>
            setCamera(
              zoomImageView(
                viewRef.current,
                [extent.width / 2, extent.height / 2],
                viewRef.current.scale * 0.8,
                size,
                extent,
              ),
            )
          }
        >
          Zoom out
        </UnstyledButton>
        <UnstyledButton type="button" disabled={disabled} onClick={resetView}>
          Reset view
        </UnstyledButton>
        <UnstyledButton
          type="button"
          disabled={disabled}
          aria-pressed={mode === 'crop'}
          onClick={() => {
            gesture.current = null;
            pointers.current.clear();
            setMode(mode === 'crop' ? 'pan' : 'crop');
          }}
        >
          Crop mode
        </UnstyledButton>
        <UnstyledButton type="button" disabled={disabled || !crop} onClick={clearCrop}>
          Cancel crop
        </UnstyledButton>
        {onFindObjects && (
          <UnstyledButton
            type="button"
            disabled={disabled || busy}
            onClick={() => onFindObjects(visibleRegion(viewRef.current, extent, size))}
          >
            Find objects
          </UnstyledButton>
        )}
        <output aria-label="Zoom level">{Math.round((view.scale / fit.scale) * 100)}%</output>
      </div>
      <fieldset
        ref={viewport}
        tabIndex={disabled ? -1 : 0}
        aria-label="Frame editor"
        aria-busy={busy}
        aria-disabled={disabled}
        data-testid="frame-editor"
        data-zoom={view.scale / fit.scale}
        data-mode={mode}
        className={classes.viewport}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
      >
        <canvas
          ref={canvas}
          className={classes.canvas}
          aria-label="Click an object to outline it"
          style={{ opacity: busy ? 0.3 : 1, pointerEvents: busy ? 'none' : 'auto' }}
          onKeyDown={event => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              event.stopPropagation();
              pick([extent.width / 2, extent.height / 2]);
            }
          }}
          onClick={event => {
            if (suppressClick.current) {
              suppressClick.current = false;
              return;
            }
            pick(local(event.clientX, event.clientY));
          }}
        />
        {crop && cropOrigin && (
          <div
            className={classes.crop}
            style={{
              left: cropOrigin[0],
              top: cropOrigin[1],
              width: crop[2] * view.scale,
              height: crop[3] * view.scale,
            }}
          >
            {mode === 'crop' && (
              <>
                <UnstyledButton
                  type="button"
                  disabled={disabled}
                  aria-label="Move crop"
                  data-crop-handle="move"
                  className={classes.move}
                />
                {HANDLES.map(handle => (
                  <UnstyledButton
                    key={handle.key}
                    type="button"
                    disabled={disabled}
                    aria-label={`Resize crop ${handle.label}`}
                    data-crop-handle={handle.key}
                    className={classes.handle}
                    style={{ left: `${handle.x * 100}%`, top: `${handle.y * 100}%`, cursor: `${handle.key}-resize` }}
                    onKeyDown={event => {
                      if (disabled || !crop) return;
                      const delta = (
                        { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] } as Record<
                          string,
                          Point
                        >
                      )[event.key];
                      if (delta) {
                        event.preventDefault();
                        event.stopPropagation();
                        onCropChange(
                          resizeFrameCrop(
                            crop,
                            handle.key,
                            [delta[0] * (event.shiftKey ? 10 : 1), delta[1] * (event.shiftKey ? 10 : 1)],
                            size,
                          ),
                        );
                      }
                    }}
                  />
                ))}
              </>
            )}
          </div>
        )}
      </fieldset>
      <p className={classes.help}>
        Wheel or pinch to zoom. Drag to pan; use Crop mode to draw or adjust a crop. Arrow keys pan, <Kbd>+</Kbd>/
        <Kbd>−</Kbd> zoom, and <Kbd>0</Kbd> resets. <Kbd>Escape</Kbd> clears the crop.
      </p>
    </fieldset>
  );
}
