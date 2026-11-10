import { fireEvent, render, screen } from '@testing-library/react';
import { type ReactElement, type ReactNode, useState } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Providers } from '../shared/Providers';
import type { Box } from './api';
import { FrameEditor } from './FrameEditor';

function EditorProviders({ children }: { children: ReactNode }) {
  return <Providers test>{children}</Providers>;
}
const renderEditor = (ui: ReactElement) => render(ui, { wrapper: EditorProviders });

class TestPointerEvent extends MouseEvent {
  pointerId: number;
  pointerType: string;
  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'mouse';
  }
}
const rect = { left: 0, top: 0, width: 320, height: 180, right: 320, bottom: 180, x: 0, y: 0, toJSON() {} };
const image = new Image();
Object.defineProperties(image, { naturalWidth: { value: 640 }, naturalHeight: { value: 360 } });
const pick = vi.fn();
const cropChanged = vi.fn();
function Harness({
  busy = false,
  disabled = false,
  selectRequest = 0,
}: {
  busy?: boolean;
  disabled?: boolean;
  selectRequest?: number;
}) {
  const [crop, setCrop] = useState<Box | null>(null);
  return (
    <>
      <FrameEditor
        image={image}
        crop={crop}
        onCropChange={box => {
          cropChanged(box);
          setCrop(box);
        }}
        onPick={pick}
        busy={busy}
        disabled={disabled}
        selectRequest={selectRequest}
      />
      <output data-testid="crop-value">{JSON.stringify(crop)}</output>
    </>
  );
}
const canvas = () => screen.getByLabelText('Click an object to outline it');
const viewport = () => screen.getByLabelText('Frame editor');
function pointer(
  type: 'pointerDown' | 'pointerMove' | 'pointerUp' | 'pointerCancel',
  target: Element,
  x: number,
  y: number,
  pointerId = 1,
  pointerType = 'mouse',
) {
  fireEvent[type](target, { clientX: x, clientY: y, pointerId, pointerType, button: 0 });
}
function drawCrop() {
  fireEvent.click(screen.getByRole('button', { name: 'Crop mode' }));
  pointer('pointerDown', canvas(), 30, 20);
  pointer('pointerMove', canvas(), 130, 70);
  pointer('pointerUp', canvas(), 130, 70);
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('PointerEvent', TestPointerEvent);
  vi.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue(rect);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    strokeRect: vi.fn(),
    setTransform: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
});

describe('FrameEditor', () => {
  it('maps a legacy click-only canvas event to full-resolution pixels', () => {
    renderEditor(<Harness />);
    fireEvent.click(canvas(), { clientX: 50, clientY: 50 });
    expect(pick).toHaveBeenCalledExactlyOnceWith(100, 100);
    expect(viewport()).toHaveAttribute('data-zoom', '1');
  });
  it('keeps wheel zoom focal and preserves original image pixels for picking', () => {
    renderEditor(<Harness />);
    fireEvent.wheel(viewport(), { clientX: 80, clientY: 45, deltaY: -Math.log(2) / 0.002 });
    expect(Number(viewport().getAttribute('data-zoom'))).toBeCloseTo(2);
    fireEvent.click(canvas(), { clientX: 80, clientY: 45 });
    expect(pick).toHaveBeenCalledExactlyOnceWith(160, 90);
  });
  it('pans a zoomed image without turning a drag into a SAM pick', () => {
    renderEditor(<Harness />);
    fireEvent.wheel(viewport(), { clientX: 160, clientY: 90, deltaY: -Math.log(2) / 0.002 });
    pointer('pointerDown', canvas(), 160, 90);
    pointer('pointerMove', canvas(), 120, 70);
    pointer('pointerUp', canvas(), 120, 70);
    fireEvent.click(canvas(), { clientX: 120, clientY: 70 });
    expect(pick).not.toHaveBeenCalled();
    fireEvent.click(canvas(), { clientX: 160, clientY: 90 });
    expect(pick).toHaveBeenCalledExactlyOnceWith(360, 200);
  });
  it('draws and resizes a full-frame crop, then Escape clears it without picking', () => {
    renderEditor(<Harness />);
    drawCrop();
    expect(cropChanged).toHaveBeenLastCalledWith([60, 40, 200, 100]);
    const right = screen.getByRole('button', { name: 'Resize crop right' });
    pointer('pointerDown', right, 130, 45);
    pointer('pointerMove', right, 500, 45);
    pointer('pointerUp', right, 500, 45);
    expect(cropChanged).toHaveBeenLastCalledWith([60, 40, 580, 100]);
    fireEvent.keyDown(screen.getByRole('button', { name: 'Resize crop left' }), { key: 'ArrowRight' });
    expect(cropChanged).toHaveBeenLastCalledWith([61, 40, 579, 100]);
    fireEvent.keyDown(screen.getByRole('button', { name: 'Resize crop left' }), { key: 'Escape' });
    expect(screen.getByTestId('crop-value')).toHaveTextContent('null');
    expect(screen.getByRole('button', { name: 'Crop mode' })).toHaveAttribute('aria-pressed', 'false');
    expect(pick).not.toHaveBeenCalled();
  });
  it('supports Cancel crop and keyboard zoom, arrows, and zero reset', () => {
    renderEditor(<Harness />);
    drawCrop();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel crop' }));
    expect(screen.getByTestId('crop-value')).toHaveTextContent('null');
    fireEvent.keyDown(viewport(), { key: '+' });
    expect(Number(viewport().getAttribute('data-zoom'))).toBeCloseTo(1.25);
    fireEvent.keyDown(viewport(), { key: 'ArrowRight' });
    fireEvent.keyDown(viewport(), { key: '0' });
    expect(viewport()).toHaveAttribute('data-zoom', '1');
  });
  it('clears the crop with Escape after a toolbar control takes focus', () => {
    renderEditor(<Harness />);
    drawCrop();
    const reset = screen.getByRole('button', { name: 'Reset view' });
    fireEvent.click(reset);
    reset.focus();
    fireEvent.keyDown(reset, { key: 'Escape' });
    expect(screen.getByTestId('crop-value')).toHaveTextContent('null');
    expect(viewport()).toHaveAttribute('data-mode', 'pan');
    expect(pick).not.toHaveBeenCalled();
  });

  it('zooms with two actual touch pointers and suppresses picks when the pinch ends', () => {
    renderEditor(<Harness />);
    pointer('pointerDown', canvas(), 80, 90, 11, 'touch');
    pointer('pointerDown', canvas(), 240, 90, 22, 'touch');
    pointer('pointerMove', canvas(), 40, 90, 11, 'touch');
    pointer('pointerMove', canvas(), 280, 90, 22, 'touch');
    expect(Number(viewport().getAttribute('data-zoom'))).toBeCloseTo(1.5);
    pointer('pointerUp', canvas(), 40, 90, 11, 'touch');
    pointer('pointerUp', canvas(), 280, 90, 22, 'touch');
    expect(pick).not.toHaveBeenCalled();
  });
  it('blocks picks while SAM is busy but keeps zoom and crop editing available', () => {
    renderEditor(<Harness busy />);
    fireEvent.click(canvas(), { clientX: 50, clientY: 50 });
    expect(pick).not.toHaveBeenCalled();
    expect(viewport()).toHaveAttribute('aria-busy', 'true');
    expect(canvas()).toHaveStyle({ opacity: '.3', pointerEvents: 'none' });
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(Number(viewport().getAttribute('data-zoom'))).toBeCloseTo(1.25);
    fireEvent.click(screen.getByRole('button', { name: 'Reset view' }));
    drawCrop();
    expect(cropChanged).toHaveBeenLastCalledWith([60, 40, 200, 100]);
  });
  it('returns to pick mode on a new selection request without losing crop or zoom', () => {
    const { rerender } = renderEditor(<Harness selectRequest={0} />);
    drawCrop();
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    const zoom = viewport().getAttribute('data-zoom');
    rerender(<Harness selectRequest={1} />);
    expect(viewport()).toHaveAttribute('data-mode', 'pan');
    expect(viewport()).toHaveAttribute('data-zoom', zoom);
    expect(screen.getByTestId('crop-value')).toHaveTextContent('[60,40,200,100]');
    fireEvent.click(canvas(), { clientX: 160, clientY: 90 });
    expect(pick).toHaveBeenCalledExactlyOnceWith(320, 180);
  });

  it('disables all gestures, keys, and controls during enrollment', () => {
    renderEditor(<Harness disabled />);
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Crop mode' })).toBeDisabled();
    fireEvent.keyDown(viewport(), { key: '+' });
    fireEvent.wheel(viewport(), { clientX: 80, clientY: 45, deltaY: -500 });
    pointer('pointerDown', canvas(), 50, 50);
    pointer('pointerMove', canvas(), 150, 100);
    pointer('pointerUp', canvas(), 150, 100);
    fireEvent.click(canvas(), { clientX: 50, clientY: 50 });
    expect(viewport()).toHaveAttribute('data-zoom', '1');
    expect(cropChanged).not.toHaveBeenCalled();
    expect(pick).not.toHaveBeenCalled();
  });
});

const findObjects = vi.fn();
const choose = vi.fn();
// Normalized outlines: a table with a cup on it.
const table: [number, number][] = [
  [0.1, 0.1],
  [0.6, 0.1],
  [0.6, 0.6],
  [0.1, 0.6],
];
const cup: [number, number][] = [
  [0.2, 0.2],
  [0.3, 0.2],
  [0.3, 0.3],
  [0.2, 0.3],
];
function Found({ busy = false }: { busy?: boolean }) {
  return (
    <FrameEditor
      image={image}
      crop={null}
      onCropChange={cropChanged}
      onPick={pick}
      busy={busy}
      onFindObjects={findObjects}
      found={[
        { id: 'table', outline: [table] },
        { id: 'cup', outline: [cup] },
      ]}
      onChoose={choose}
    />
  );
}

describe('Find objects (#2020)', () => {
  it('asks for the whole frame at 1x and only the region in view when zoomed', () => {
    renderEditor(<Found />);
    fireEvent.click(screen.getByRole('button', { name: 'Find objects' }));
    expect(findObjects).toHaveBeenLastCalledWith([0, 0, 640, 360]);
    // 2x around (80, 45) on the 320 x 180 canvas: frame (160, 90) stays put, so the view starts at (80, 45).
    fireEvent.wheel(viewport(), { clientX: 80, clientY: 45, deltaY: -Math.log(2) / 0.002 });
    fireEvent.click(screen.getByRole('button', { name: 'Find objects' }));
    expect(findObjects).toHaveBeenLastCalledWith([80, 45, 320, 180]);
  });

  it('picks the smallest outline under a click, and a click outside every outline outlines a point', () => {
    renderEditor(<Found />);
    fireEvent.click(canvas(), { clientX: 80, clientY: 45 }); // frame (160, 90): normalized (0.25, 0.25), in both
    expect(choose).toHaveBeenLastCalledWith('cup');
    fireEvent.click(canvas(), { clientX: 160, clientY: 72 }); // (0.5, 0.4): the table only
    expect(choose).toHaveBeenLastCalledWith('table');
    expect(pick).not.toHaveBeenCalled();
    fireEvent.click(canvas(), { clientX: 250, clientY: 150 }); // (0.78, 0.83): neither
    expect(pick).toHaveBeenCalledExactlyOnceWith(500, 300);
    expect(choose).toHaveBeenCalledTimes(2);
  });

  it('is disabled while EdgeTAM is working', () => {
    renderEditor(<Found busy />);
    expect(screen.getByRole('button', { name: 'Find objects' })).toBeDisabled();
  });
});
