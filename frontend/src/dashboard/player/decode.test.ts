import * as flatbuffers from 'flatbuffers';
import { describe, expect, it } from 'vitest';
import { sort } from './decode';
import { Box, Caption, Joint, Overlay, Skeleton } from './generated/continuum/overlay.js';
import { encodeHeader, HEADER_LEN, PayloadType } from './header';

/** Builds an overlay the way the Rust gateway does, so decoding is tested against real FlatBuffer bytes. */
function overlayBytes(captureUs: number, tracked = false): Uint8Array {
  const b = new flatbuffers.Builder(256);
  const label = b.createString('person');
  const kind = b.createString('face');
  const id = b.createString('r:cam0:1');
  const box = Box.createBox(b, 0.1, 0.2, 0.3, 0.4, 0.9, label, 120, kind, id);
  const boxes = Overlay.createBoxesVector(b, [box]);
  const sbox = Box.createBox(
    b,
    0,
    0,
    1,
    1,
    0.5,
    b.createString(''),
    tracked ? 211 : -1,
    0,
    tracked ? b.createString('run:cam0:8') : 0,
  );
  const layout = b.createString('halpe26');
  Skeleton.startJointsVector(b, 2);
  Joint.createJoint(b, -1, -1, 0); // vectors of structs are built back to front
  Joint.createJoint(b, 0.5, 0.5, 0.9);
  const joints = b.endVector();
  const skeleton = Skeleton.createSkeleton(b, sbox, layout, joints);
  const skeletons = Overlay.createSkeletonsVector(b, [skeleton]);
  const caption = Caption.createCaption(b, b.createString('hello'), b.createString('Sam'));
  const captions = Overlay.createCaptionsVector(b, [caption]);
  Overlay.startOverlay(b);
  Overlay.addCaptureUs(b, BigInt(captureUs));
  Overlay.addBoxes(b, boxes);
  Overlay.addSkeletons(b, skeletons);
  Overlay.addCaptions(b, captions);
  b.finish(Overlay.endOverlay(b));
  return b.asUint8Array();
}

function message(type: number, captureUs: number, payload: Uint8Array): ArrayBuffer {
  const out = new Uint8Array(HEADER_LEN + payload.length);
  out.set(
    encodeHeader({ streamId: 4, payloadType: type as 1, keyframe: type === PayloadType.jpeg, live: false, captureUs }),
  );
  out.set(payload, HEADER_LEN);
  return out.buffer;
}

describe('sort', () => {
  it('decodes an overlay payload into plain boxes, skeletons, and captions', () => {
    const got = sort(message(PayloadType.overlay, 777, overlayBytes(777)));
    expect(got.kind).toBe('overlay');
    if (got.kind !== 'overlay') return;
    expect(got.overlay.captureUs).toBe(777);
    expect(got.overlay.boxes[0]).toMatchObject({
      score: expect.closeTo(0.9, 5),
      label: 'person',
      hue: 120,
      kind: 'face',
      id: 'r:cam0:1',
    });
    expect(got.overlay.boxes[0]?.xyxy.map(v => Math.round(v * 10))).toEqual([1, 2, 3, 4]);
    expect(got.overlay.skeletons[0]?.layout).toBe('halpe26');
    expect(got.overlay.skeletons[0]?.joints.map(j => j.map(v => Math.round(v * 10)))).toEqual([
      [5, 5, 9],
      [-10, -10, 0],
    ]);
    expect(got.overlay.skeletons[0]?.box.hue).toBeNull();
    expect(got.overlay.skeletons[0]?.box.kind).toBeUndefined();
    expect(got.overlay.captions).toEqual([{ text: 'hello', speaker: 'Sam' }]);
  });

  it('retains a skeleton track and hue through real FlatBuffer bytes', () => {
    const got = sort(message(PayloadType.overlay, 777, overlayBytes(777, true)));
    expect(got.kind).toBe('overlay');
    if (got.kind === 'overlay') expect(got.overlay.skeletons[0]?.box).toMatchObject({ id: 'run:cam0:8', hue: 211 });
  });

  it('falls back to the header time when the table has none', () => {
    const got = sort(message(PayloadType.overlay, 55, overlayBytes(0)));
    if (got.kind === 'overlay') expect(got.overlay.captureUs).toBe(55);
  });

  it('passes video, jpeg, and arrow payloads through with their headers', () => {
    const jpeg = sort(message(PayloadType.jpeg, 9, new Uint8Array([0xff, 0xd8])));
    expect(jpeg).toMatchObject({ kind: 'jpeg', header: { streamId: 4, keyframe: true, live: false, captureUs: 9 } });
    expect(sort(message(PayloadType.video, 1, new Uint8Array([0, 0, 1]))).kind).toBe('video');
    expect(sort(message(PayloadType.arrow, 1, new Uint8Array([65]))).kind).toBe('arrow');
  });
});
