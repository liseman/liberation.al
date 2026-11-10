// A selected face from Live or Review stays in this tab as evidence, never as an inferred personal name.
import { request } from '../../shared/http';
import type { BoxPick } from '../camera/boxes';
import { nameLiveFace } from '../camera/gateway';
export interface FaceSource {
  camera: string;
  at: number;
  track: string;
  group: number;
}
export interface LiveFaceDraft {
  source: FaceSource;
  image: string;
}
let pending: LiveFaceDraft | null = null;
export function prepareLiveFace(pick: BoxPick, canvas: HTMLCanvasElement): LiveFaceDraft {
  if (!pick.faceSource || !Number.isFinite(pick.faceSource.at))
    throw new Error('This face has no stable reference yet. Try again.');
  const [x1, y1, x2, y2] = pick.box as [number, number, number, number];
  if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1 || !canvas.width || !canvas.height)
    throw new Error('No selected face picture is available.');
  const x = Math.max(0, x1) * canvas.width,
    y = Math.max(0, y1) * canvas.height;
  const width = Math.min(canvas.width, Math.max(0, x2) * canvas.width) - x;
  const height = Math.min(canvas.height, Math.max(0, y2) * canvas.height) - y;
  if (width < 2 || height < 2) throw new Error('The selected face is too small.');
  const crop = document.createElement('canvas'),
    scale = Math.min(1, 512 / Math.max(width, height));
  crop.width = Math.round(width * scale);
  crop.height = Math.round(height * scale);
  const context = crop.getContext('2d');
  if (!context) throw new Error('Could not prepare the face picture.');
  context.drawImage(canvas, x, y, width, height, 0, 0, crop.width, crop.height);
  const image = crop.toDataURL('image/jpeg', 0.85);
  if (!image.startsWith('data:image/jpeg;base64,') || image.length > 1_000_000)
    throw new Error('Could not prepare the face picture.');
  return { source: { ...pick.faceSource }, image };
}
export function handoffLiveFace(draft: LiveFaceDraft): void {
  pending = draft;
}
export function takeLiveFace(): LiveFaceDraft | null {
  const result = pending;
  pending = null;
  return result;
}
export const claimLiveFace = async (draft: LiveFaceDraft, name: string) => {
  const reply = await request<{ name: string }>('/people/claim', {
    kind: 'face',
    id: draft.source.group,
    name,
    face_source: draft.source,
    face_image: draft.image,
  });
  nameLiveFace(draft.source, reply.name);
  return reply;
};
