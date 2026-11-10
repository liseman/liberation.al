// Body skeletons (#520, and Review's #519): the halpe26 keypoint layout vision's pose models give (vision/pose_model.py,
// see/actions.py, `JOINTS`), which joints a bone joins, and the rule for what's drawn: a joint, and a bone, only where the
// model is sure enough. Shared by Live (over the live camera) and Review (over recorded video, from `pose.track`).
/** The 26 joints, in the order the keypoints come. */
// biome-ignore format: one list
export const HALPE26 = [
  'nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow',
  'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle',
  'right ankle', 'head', 'neck', 'hip', 'left big toe', 'right big toe', 'left small toe', 'right small toe',
  'left heel', 'right heel',
] as const;

/** The wireframe's bones, as pairs of joint indexes (halpe26's skeleton, as see-actions' poses use it). */
// biome-ignore format: one list
export const BONES: readonly (readonly [number, number])[] = [
  [15, 13], [13, 11], [11, 19], [16, 14], [14, 12], [12, 19], [17, 18], [18, 19], [18, 5], [5, 7], [7, 9], [18, 6],
  [6, 8], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [15, 20], [15, 22], [15, 24], [16, 21], [16, 23],
  [16, 25],
];

/** A joint is drawn at this confidence or above, and a bone only when both its joints are. On devbox's cameras the
 *  head and shoulders come in around 0.7 to 0.9, and legs out of frame around 0.1 to 0.2. It is see-actions' sure
 *  joint (actions.JOINT_SURE), so what's drawn is what the tracker and the action models trust (#2024). */
export const MIN_CONF = 0.5;

/** One person's pose as vision sends it: box [x1, y1, x2, y2, score] and keypoints [[x, y, confidence], ...] in
 *  joint order, all 0 to 1 of the frame; a joint the model didn't place is [-1, -1, 0]. */
export interface Pose {
  /** Camera-local track, absent for old recordings or untracked hints. */
  track_id?: string;
  /** Stable gateway hue for this track, absent for old recordings. */
  hue?: number | null;
  box: number[];
  keypoints: number[][];
}

/** What to draw for one person: the bones as [x1, y1, x2, y2] and the joints as [x, y], 0 to 1 of the frame. */
export function wireframe(p: Pose, minConf = MIN_CONF): { bones: number[][]; joints: number[][] } {
  const seen = p.keypoints.map(k => k.length >= 3 && (k[2] ?? 0) >= minConf && (k[0] ?? -1) >= 0 && (k[1] ?? -1) >= 0);
  const joints = p.keypoints.filter((_, i) => seen[i]).map(k => [k[0] ?? 0, k[1] ?? 0]);
  const bones = BONES.filter(([a, b]) => seen[a] && seen[b]).map(([a, b]) => {
    const [ka = [], kb = []] = [p.keypoints[a], p.keypoints[b]];
    return [ka[0] ?? 0, ka[1] ?? 0, kb[0] ?? 0, kb[1] ?? 0];
  });
  return { bones, joints };
}
