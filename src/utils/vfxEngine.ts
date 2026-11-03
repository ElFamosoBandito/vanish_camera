export type VanishMode = 'normal' | 'naruto' | 'goku';

export type AppPhase =
  | 'calibrating'
  | 'ready'
  | 'vanishing'
  | 'vanished'
  | 'reappearing';

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface ModeDefinition {
  id: VanishMode;
  key: string;
  title: string;
  gestureTitle: string;
  description: string;
  durationSec: number;
}

export const CALIBRATION_SECONDS = 4.0;
export const BACKGROUND_SAMPLE_SECONDS = 0.55;
export const VANISH_SECONDS = 1.15;
export const NARUTO_EFFECT_SECONDS = 1.45;
export const GESTURE_INTERVAL_SECONDS = 0.07;

export const VANISH_MODES: ModeDefinition[] = [
  {
    id: 'normal',
    key: '1',
    title: '1 · Normal Vanish (Fade Out)',
    gestureTitle: 'Signe V (Index + Majeur écartés)',
    description:
      'Fondu progressif uniforme (Fade Out) de votre silhouette vers le fond calibré.',
    durationSec: 1.15,
  },
  {
    id: 'naruto',
    key: '2',
    title: '2 · Naruto Vanish (Substitution)',
    gestureTitle: 'Signe Naruto (Mains jointes / Mudra)',
    description:
      'Portage exact de naruto_effect.py : effacement en 0.13s, double couche de fumée et chute de votre bûche avec rebond.',
    durationSec: 1.45,
  },
  {
    id: 'goku',
    key: '3',
    title: '3 · Téléportation Goku (Shunkan Idō)',
    gestureTitle: 'Index + Majeur posés sur le front',
    description:
      'Style anime DBZ authentique : hachures verticales noires/blanches et découpe en bandes verticales révélant le fond.',
    durationSec: 1.15,
  },
];

/**
 * Deterministic Mulberry32 PRNG for reproducible noise fields & Naruto smoke layers
 */
export function createSeededRng(seed: number): () => number {
  let a = seed | 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function smoothstepScalar(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3.0 - 2.0 * t);
}

export function easeInOutCubic(value: number): number {
  if (value < 0.5) {
    return 4.0 * value * value * value;
  }
  return 1.0 - Math.pow(-2.0 * value + 2.0, 3) / 2.0;
}

/**
 * Exact port of `stack = np.stack(background_samples[-18:], axis=0); background = np.median(stack, axis=0).astype(np.uint8)`
 * from `Nicovic17/vanish_webcam/vanish_cam.py`.
 * Computes the pixel-wise median across up to 18 frames captured during the last 0.55s of the 4.0s calibration.
 */
export function computeMedianBackground(
  samples: Uint8ClampedArray[],
  outBg: Uint8ClampedArray
): void {
  if (samples.length === 0) return;
  const recent = samples.slice(-18);
  const count = recent.length;
  if (count === 1) {
    outBg.set(recent[0]);
    return;
  }

  const totalBytes = outBg.length;
  const mid = count >> 1;
  const vals = new Uint8Array(count);

  for (let i = 0; i < totalBytes; i += 4) {
    for (let c = 0; c < 3; c++) {
      const idx = i + c;
      for (let s = 0; s < count; s++) {
        vals[s] = recent[s][idx];
      }
      vals.sort();
      outBg[idx] =
        count % 2 === 1 ? vals[mid] : (vals[mid - 1] + vals[mid]) >> 1;
    }
    outBg[i + 3] = 255;
  }
}

/**
 * Port of `make_dissolve_fields` from `Nicovic17/vanish_webcam/vanish_cam.py`.
 * Generates the 2D noise dissolve field and sparkle noise array for a given resolution.
 */
export function makeDissolveFields(
  width: number,
  height: number
): { field: Float32Array; sparkleNoise: Float32Array } {
  const rng = createSeededRng(7391);
  const gridW = Math.max(2, Math.floor(width / 10));
  const gridH = Math.max(2, Math.floor(height / 10));
  const lowGrid = new Float32Array(gridW * gridH);
  for (let i = 0; i < lowGrid.length; i++) {
    lowGrid[i] = rng();
  }

  const field = new Float32Array(width * height);
  const sparkleNoise = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    const yNorm = y / Math.max(1, height - 1);
    const gy = yNorm * (gridH - 1);
    const y0 = Math.floor(gy);
    const y1 = Math.min(gridH - 1, y0 + 1);
    const wy = gy - y0;

    for (let x = 0; x < width; x++) {
      const xNorm = x / Math.max(1, width - 1);
      const gx = xNorm * (gridW - 1);
      const x0 = Math.floor(gx);
      const x1 = Math.min(gridW - 1, x0 + 1);
      const wx = gx - x0;

      // Bilinear interpolation of low-frequency noise
      const top = lowGrid[y0 * gridW + x0] * (1 - wx) + lowGrid[y0 * gridW + x1] * wx;
      const bottom = lowGrid[y1 * gridW + x0] * (1 - wx) + lowGrid[y1 * gridW + x1] * wx;
      const lowNoise = top * (1 - wy) + bottom * wy;

      const fine = rng();
      const idx = y * width + x;
      sparkleNoise[idx] = fine;

      const val =
        (1.0 - yNorm) * 0.72 +
        lowNoise * 0.17 +
        fine * 0.08 +
        Math.sin(xNorm * 18.0) * 0.025;
      field[idx] = Math.max(0.0, Math.min(1.0, val));
    }
  }

  return { field, sparkleNoise };
}

/**
 * Port of `substitution_anchor(mask)` from `Nicovic17/vanish_webcam/naruto_effect.py`.
 * Finds the person's torso bounding center and scale from the segmentation mask.
 */
export function substitutionAnchor(
  mask: Float32Array,
  width: number,
  height: number
): [number, number, number] {
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  let count = 0;

  for (let y = 0; y < height; y += 2) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 2) {
      if (mask[rowOffset + x] > 0.45) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        count++;
      }
    }
  }

  if (count < 25 || maxX <= minX || maxY <= minY) {
    return [Math.floor(width / 2), Math.floor(height / 2), Math.floor(height * 0.57)];
  }

  const w = maxX - minX;
  const h = maxY - minY;
  const size = Math.floor(Math.max(height * 0.24, Math.min(height * 0.7, h * 0.58)));
  const margin = Math.floor(size / 3);
  const centerX = Math.floor(Math.max(margin, Math.min(width - margin, minX + w * 0.5)));
  const centerY = Math.floor(Math.max(margin, Math.min(height - margin, minY + h * 0.52)));

  return [centerX, centerY, size];
}

/**
 * Port of `landed_log_anchor` & `falling_log_anchor` from `Nicovic17/vanish_webcam/naruto_effect.py`.
 * Accelerates downwards under gravity, touches the frame floor, then settles with a damped bounce.
 */
export function landedLogAnchor(
  anchor: [number, number, number],
  frameHeight: number
): [number, number, number] {
  const [cx, cy, size] = anchor;
  return [cx, Math.max(cy, frameHeight - Math.round(size * 0.48)), size];
}

export function fallingLogAnchor(
  anchor: [number, number, number],
  frameHeight: number,
  age: number
): [number, number, number] {
  const [cx, cy, size] = anchor;
  const floorY = landedLogAnchor(anchor, frameHeight)[1];
  const distance = floorY - cy;
  if (distance <= 0 || age <= 0.18) {
    return anchor;
  }
  const elapsed = age - 0.18;
  const gravity = frameHeight * 3.2;
  const fall = gravity * elapsed * elapsed * 0.5;
  if (fall < distance) {
    return [cx, cy + Math.round(fall), size];
  }
  const landedAt = Math.sqrt((2.0 * distance) / gravity);
  const bounceAge = elapsed - landedAt;
  const bounce =
    size * 0.075 * Math.exp(-5.0 * bounceAge) * Math.abs(Math.sin(11.0 * bounceAge));
  return [cx, floorY - Math.round(bounce), size];
}

/**
 * Port of `victory_landmark_score` from `Nicovic17/vanish_webcam/vanish_cam.py`.
 * Recognizes index and middle fingers extended in a V-sign while ring and pinky are folded,
 * invariant to 3D hand rotation and scale.
 */
export function victoryLandmarkScore(points: Point3D[]): number {
  if (!points || points.length !== 21) return 0.0;
  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
      return 0.0;
    }
  }

  const dist3D = (a: Point3D, b: Point3D) =>
    Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

  const straightness: number[] = [];
  const extended: boolean[] = [];
  const folded: boolean[] = [];
  const lengths: number[] = [];

  const bases = [5, 9, 13, 17];
  for (let index = 0; index < 4; index++) {
    const base = bases[index];
    const f0 = points[base];
    const f1 = points[base + 1];
    const f2 = points[base + 2];
    const f3 = points[base + 3];

    const length = dist3D(f0, f1) + dist3D(f1, f2) + dist3D(f2, f3);
    if (length < 1e-6) {
      if (index < 2) return 0.0;
      straightness.push(0.0);
      lengths.push(0.0);
      extended.push(false);
      folded.push(true);
      continue;
    }

    const ratio = dist3D(f3, f0) / length;
    const tipDistance = dist3D(f3, points[0]);
    const jointDistance = dist3D(f1, points[0]);

    straightness.push(ratio);
    lengths.push(length);
    extended.push(ratio >= 0.82 && tipDistance > jointDistance * 1.1);
    folded.push(ratio < 0.78 || tipDistance <= jointDistance * 1.05);
  }

  const referenceLength = Math.min(lengths[0], lengths[1]);
  for (const [index, base] of [
    [2, 13],
    [3, 17],
  ] as const) {
    const reach = dist3D(points[base + 3], points[base]);
    folded[index] = folded[index] || reach < referenceLength * 0.65;
  }

  if (!(extended[0] && extended[1] && folded[2] && folded[3])) {
    return 0.0;
  }

  const separation = dist3D(points[8], points[12]);
  const mcpSeparation = dist3D(points[5], points[9]);
  const dipSeparation = dist3D(points[7], points[11]);

  // Clear "V" spread requirement: Index (8) and Middle (12) must diverge into a V shape
  // (tips significantly wider apart than base knuckles and DIP joints), never glued together
  if (
    separation < referenceLength * 0.32 ||
    separation <= mcpSeparation * 1.18 ||
    separation <= dipSeparation * 1.02
  ) {
    return 0.0;
  }

  return Math.max(0.0, Math.min(1.0, Math.min(straightness[0], straightness[1])));
}

/**
 * Exact port of `classify_gesture_result` from `Nicovic17/vanish_webcam/vanish_cam.py` (lines 433-450):
 * Checks both 3D `hand_world_landmarks` AND 2D `hand_landmarks` projected to `(x * width, y * height, 0.0)`.
 */
export function classifyVictoryFromHands(
  worldHands: Point3D[][],
  imageHands: Point2D[][],
  width: number,
  height: number
): number {
  if (worldHands) {
    for (const landmarks of worldHands) {
      const score = victoryLandmarkScore(landmarks);
      if (score > 0) return score;
    }
  }
  if (imageHands) {
    for (const landmarks of imageHands) {
      const points3D: Point3D[] = landmarks.map((p) => ({
        x: p.x * width,
        y: p.y * height,
        z: 0.0,
      }));
      const score = victoryLandmarkScore(points3D);
      if (score > 0) return score;
    }
  }
  return 0.0;
}

/**
 * Helper: verifies whether a hand (in 2D pixel space) forms the vertical Naruto Tiger/Ram Mudra blade:
 * - Index (8) and Middle (12) are extended straight up beyond their PIP (6, 10) and MCP (5, 9) joints
 * - Index (8) and Middle (12) are pressed tightly side-by-side (collinear blade, NOT spread like a V-sign)
 * - Ring (16) and Pinky (20) are folded/interlocked downward into the palm
 */
function hasNarutoMudraFingerPosture(
  h: Point2D[],
  scale: number,
  strictSingleMergedHand: boolean = false
): boolean {
  const dist2D = (a: Point2D, b: Point2D) => Math.hypot(a.x - b.x, a.y - b.y);

  const indexLen = dist2D(h[0], h[8]);
  const middleLen = dist2D(h[0], h[12]);
  const indexPipLen = dist2D(h[0], h[6]);
  const middlePipLen = dist2D(h[0], h[10]);

  // 1. Index & Middle must be genuinely extended upward
  const indexExtended =
    indexLen > indexPipLen * 1.08 &&
    indexLen > scale * 1.15 &&
    h[8].y < h[6].y;
  const middleExtended =
    middleLen > middlePipLen * 1.08 &&
    middleLen > scale * 1.15 &&
    (h[12].y < h[10].y || dist2D(h[8], h[12]) < scale * 0.45);

  if (!indexExtended || !middleExtended) return false;

  // 2. Index & Middle must point vertically upward (wrist below fingertips)
  const verticalReach =
    h[0].y - h[8].y > scale * (strictSingleMergedHand ? 0.75 : 0.65) &&
    h[0].y - h[12].y > scale * (strictSingleMergedHand ? 0.75 : 0.65);
  if (!verticalReach) return false;

  if (strictSingleMergedHand) {
    // When holding in front of face/chest, allow slight natural tilt up to ~37°
    const indexVertical = Math.abs(h[8].x - h[5].x) <= Math.max(1, h[5].y - h[8].y) * 0.75;
    const middleVertical = Math.abs(h[12].x - h[9].x) <= Math.max(1, h[9].y - h[12].y) * 0.75;
    if (!indexVertical || !middleVertical) return false;
  }

  // 3. Index (8) and Middle (12) must be pressed tightly together (ZERO V-sign spread)
  const maxTipGap = scale * (strictSingleMergedHand ? 0.58 : 0.65);
  const tipsTogether = dist2D(h[8], h[12]) < maxTipGap;
  const notSpreadV = dist2D(h[8], h[12]) <= Math.max(dist2D(h[5], h[9]) * 1.35, scale * 0.52);
  if (!tipsTogether || !notSpreadV) return false;

  // 4. Ring (16) and Pinky (20) must be folded / interlocked down (rejects open palms)
  const ringLen = dist2D(h[0], h[16]);
  const pinkyLen = dist2D(h[0], h[20]);
  const bladeTopY = Math.min(h[8].y, h[12].y);

  const ringFolded =
    h[16].y > bladeTopY + scale * 0.20 ||
    ringLen < indexLen * 0.90 ||
    ringLen < dist2D(h[0], h[14]) * 1.15;
  const pinkyFolded =
    h[20].y > bladeTopY + scale * 0.20 ||
    pinkyLen < middleLen * 0.90 ||
    pinkyLen < dist2D(h[0], h[18]) * 1.15;

  return ringFolded && pinkyFolded;
}

/**
 * Recognizes two nearby, vertically aligned hands clasped together in front of the chest
 * in the Naruto mudra posture (rejecting open palms or separated hands).
 */
export function joinedHandsFromLandmarks(
  hands: Point2D[][],
  width: number,
  height: number
): boolean {
  if (!hands || hands.length < 2) return false;
  const validHands = hands.slice(0, 2).filter((h) => h && h.length === 21);
  if (validHands.length !== 2) return false;

  const pts = validHands.map((hand) =>
    hand.map((p) => ({ x: p.x * width, y: p.y * height }))
  );

  for (const hand of pts) {
    for (const p of hand) {
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return false;
    }
  }

  const dist2D = (a: Point2D, b: Point2D) => Math.hypot(a.x - b.x, a.y - b.y);
  const palmIndices = [0, 5, 9, 13, 17];
  const tipIndices = [4, 8, 12, 16, 20];

  const palms = pts.map((hand) => ({
    x: palmIndices.reduce((s, idx) => s + hand[idx].x, 0) / palmIndices.length,
    y: palmIndices.reduce((s, idx) => s + hand[idx].y, 0) / palmIndices.length,
  }));

  const fingertips = pts.map((hand) => ({
    x: tipIndices.reduce((s, idx) => s + hand[idx].x, 0) / tipIndices.length,
    y: tipIndices.reduce((s, idx) => s + hand[idx].y, 0) / tipIndices.length,
  }));

  // Hands must be in the torso/chin zone
  for (const palm of palms) {
    if (
      palm.x < width * 0.14 ||
      palm.x > width * 0.86 ||
      palm.y < height * 0.16 ||
      palm.y > height * 0.88
    ) {
      return false;
    }
  }

  const scales = pts.map((hand) =>
    Math.max(dist2D(hand[5], hand[17]), dist2D(hand[0], hand[9]))
  );
  const minScale = Math.min(scales[0], scales[1]);
  const maxScale = Math.max(scales[0], scales[1]);
  if (minScale < 12.0 || maxScale > minScale * 2.1) return false;

  const size = (scales[0] + scales[1]) * 0.5;

  // Both hands must be oriented upward (wrist below middle knuckle)
  const up0 = pts[0][0].y - pts[0][9].y > scales[0] * 0.35;
  const up1 = pts[1][0].y - pts[1][9].y > scales[1] * 0.35;
  if (!up0 || !up1) return false;

  // Palms or fingers must be clasped close together
  const palmsJoined = dist2D(palms[0], palms[1]) < size * 1.65;
  const tipsJoined =
    dist2D(fingertips[0], fingertips[1]) < size * 1.85 ||
    dist2D(pts[0][8], pts[1][8]) < size * 1.15 ||
    dist2D(pts[0][12], pts[1][12]) < size * 1.15 ||
    dist2D(pts[0][5], pts[1][5]) < size * 1.25;

  if (!palmsJoined || !tipsJoined) return false;

  // Because one hand wraps around the other in the interlocked Naruto Tiger/Ram seal,
  // at least ONE hand must show the Naruto Mudra posture (Index+Middle up & together, Ring+Pinky folded),
  // AND neither hand can be a wide-open 5-finger palm (Ring & Pinky must be lower than Index & Middle).
  const mudra0 = hasNarutoMudraFingerPosture(pts[0], scales[0], false);
  const mudra1 = hasNarutoMudraFingerPosture(pts[1], scales[1], false);

  const notOpenPalm0 =
    pts[0][16].y > Math.min(pts[0][8].y, pts[0][12].y) + scales[0] * 0.22 &&
    pts[0][20].y > Math.min(pts[0][8].y, pts[0][12].y) + scales[0] * 0.22;
  const notOpenPalm1 =
    pts[1][16].y > Math.min(pts[1][8].y, pts[1][12].y) + scales[1] * 0.22 &&
    pts[1][20].y > Math.min(pts[1][8].y, pts[1][12].y) + scales[1] * 0.22;

  return (mudra0 || mudra1) && notOpenPalm0 && notOpenPalm1;
}

/**
 * Recognizes the Naruto Substitution Hand Seal (from the user's 3 reference screenshots):
 * - Never triggers if the hand is on the forehead (`isGokuForeheadGesture`).
 * - Case 1 (MediaPipe detects 2 hands): Both hands are clasped/crossed in front of the chest/chin
 *   with Index (8) + Middle (12) raised together and Ring (16) + Pinky (20) folded/interlocked.
 * - Case 2 (MediaPipe merges the 2 tightly interlocked hands into 1 hand):
 *   The interlocked Tiger-seal hand is held in the central chest/chin zone (well below the forehead/eyes),
 *   with Index (8) + Middle (12) standing straight vertically up and glued tightly together,
 *   and Ring (16) + Pinky (20) tightly folded down.
 */
export function isNarutoHandSeal(
  hands: Point2D[][],
  width: number,
  height: number,
  faceForeheadPoint: Point2D | null = null
): boolean {
  if (!hands || hands.length === 0) return false;
  if (isGokuForeheadGesture(hands, faceForeheadPoint)) return false;

  if (hands.length >= 2 && joinedHandsFromLandmarks(hands, width, height)) {
    return true;
  }

  const dist2D = (a: Point2D, b: Point2D) => Math.hypot(a.x - b.x, a.y - b.y);
  const validHands = hands
    .filter((h) => h && h.length === 21)
    .map((hand) => hand.map((p) => ({ x: p.x * width, y: p.y * height })));

  if (validHands.length === 0) return false;
  const palmIndices = [0, 5, 9, 13, 17];

  // Case 1B: 2 hands detected where the 2-finger mudra blades cross or touch
  if (validHands.length >= 2) {
    const h0 = validHands[0];
    const h1 = validHands[1];
    const palm0 = {
      x: palmIndices.reduce((s, idx) => s + h0[idx].x, 0) / palmIndices.length,
      y: palmIndices.reduce((s, idx) => s + h0[idx].y, 0) / palmIndices.length,
    };
    const palm1 = {
      x: palmIndices.reduce((s, idx) => s + h1[idx].x, 0) / palmIndices.length,
      y: palmIndices.reduce((s, idx) => s + h1[idx].y, 0) / palmIndices.length,
    };

    const scale0 = Math.max(dist2D(h0[5], h0[17]), dist2D(h0[0], h0[9]));
    const scale1 = Math.max(dist2D(h1[5], h1[17]), dist2D(h1[0], h1[9]));
    const size = (scale0 + scale1) * 0.5;

    const blade0 = hasNarutoMudraFingerPosture(h0, scale0, false);
    const blade1 = hasNarutoMudraFingerPosture(h1, scale1, false);

    if (blade0 || blade1) {
      const handsClose =
        dist2D(palm0, palm1) < size * 1.75 ||
        dist2D(h0[8], h1[8]) < size * 1.15 ||
        dist2D(h0[7], h1[7]) < size * 1.10 ||
        dist2D(h0[12], h1[12]) < size * 1.15;
      if (handsClose) return true;
    }
  }

  // Case 2: MediaPipe merged the two tightly interlocked Tiger-seal hands into 1 detected hand
  // Must be in the chest/chin zone (well below the forehead) with strict vertical Tiger-seal posture
  for (const h of validHands) {
    const scale = Math.max(dist2D(h[5], h[17]), dist2D(h[0], h[9]));
    if (scale < 14.0) continue;

    const palm = {
      x: palmIndices.reduce((s, idx) => s + h[idx].x, 0) / palmIndices.length,
      y: palmIndices.reduce((s, idx) => s + h[idx].y, 0) / palmIndices.length,
    };
    const midTip = {
      x: (h[8].x + h[12].x) * 0.5,
      y: (h[8].y + h[12].y) * 0.5,
    };

    // In front of torso / face / chin (allows natural positions across frame)
    if (
      palm.x < width * 0.12 ||
      palm.x > width * 0.88 ||
      midTip.x < width * 0.12 ||
      midTip.x > width * 0.88
    ) {
      continue;
    }

    // Must be below forehead (nose/mouth/chin/chest zone, exactly matching user reference photo)
    const minChestTipY = faceForeheadPoint
      ? (faceForeheadPoint.y + 0.04) * height
      : height * 0.18;
    const minChestPalmY = faceForeheadPoint
      ? (faceForeheadPoint.y + 0.12) * height
      : height * 0.28;

    if (midTip.y < minChestTipY || palm.y < minChestPalmY || palm.y > height * 0.95) {
      continue;
    }

    if (hasNarutoMudraFingerPosture(h, scale, true)) {
      return true;
    }
  }

  return false;
}

/**
 * Detects Goku's Instant Transmission (Shunkan Idō) pose:
 * - Hand closed (Ring 16 and Pinky 20 folded)
 * - Index (8) and Middle (12) standing up (`debout`) and pressed together (NOT spread in a V)
 * - Index (8) and Middle (12) touching the forehead / upper head region
 */
export function isGokuForeheadGesture(
  hands: Point2D[][],
  faceForeheadPoint: Point2D | null
): boolean {
  if (!hands || hands.length === 0) return false;

  for (const hand of hands) {
    if (!hand || hand.length < 21) continue;
    const indexTip = hand[8];
    const middleTip = hand[12];
    const midPoint = {
      x: (indexTip.x + middleTip.x) * 0.5,
      y: (indexTip.y + middleTip.y) * 0.5,
    };

    const scale = Math.max(
      0.04,
      Math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y),
      Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y)
    );

    // 1. Index (8) and Middle (12) must be standing up (`index et majeur debout`)
    const fingersStandingUp =
      indexTip.y < hand[6].y &&
      middleTip.y < hand[10].y &&
      hand[0].y > midPoint.y + scale * 0.55;
    if (!fingersStandingUp) continue;

    // 2. Index (8) and Middle (12) must be pressed together (NOT spread in a V)
    const tipGap = Math.hypot(indexTip.x - middleTip.x, indexTip.y - middleTip.y);
    const mcpGap = Math.hypot(hand[5].x - hand[9].x, hand[5].y - hand[9].y);
    const tipsTogether = tipGap < Math.max(0.095, scale * 0.55) && tipGap <= mcpGap * 1.25;
    if (!tipsTogether) continue;

    // 3. Rest of the hand closed (`main fermée` — Ring 16 and Pinky 20 folded down)
    const indexReach = Math.hypot(indexTip.x - hand[0].x, indexTip.y - hand[0].y);
    const middleReach = Math.hypot(middleTip.x - hand[0].x, middleTip.y - hand[0].y);
    const ringReach = Math.hypot(hand[16].x - hand[0].x, hand[16].y - hand[0].y);
    const pinkyReach = Math.hypot(hand[20].x - hand[0].x, hand[20].y - hand[0].y);

    const ringFolded =
      hand[16].y > midPoint.y + scale * 0.18 || ringReach < indexReach * 0.88;
    const pinkyFolded =
      hand[20].y > midPoint.y + scale * 0.18 || pinkyReach < middleReach * 0.88;
    if (!ringFolded || !pinkyFolded) continue;

    // 4. Fingertips must touch the forehead / upper head zone
    if (faceForeheadPoint) {
      const distToForehead = Math.hypot(
        midPoint.x - faceForeheadPoint.x,
        midPoint.y - faceForeheadPoint.y
      );
      const onForeheadHeight =
        midPoint.y <= faceForeheadPoint.y + 0.115 &&
        midPoint.y >= faceForeheadPoint.y - 0.14;
      if (distToForehead < 0.155 && onForeheadHeight) {
        return true;
      }
    } else {
      // Fallback when the hand placed on the forehead partially occludes FaceLandmarker:
      const inUpperHeadZone =
        midPoint.x >= 0.25 &&
        midPoint.x <= 0.75 &&
        midPoint.y >= 0.05 &&
        midPoint.y <= 0.32;
      if (inUpperHeadZone) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Evaluates all 3 vanish gestures simultaneously so the user never has to manually select a mode:
 * - Goku ('goku'): Closed hand with Index + Middle standing up & touching the forehead
 * - Normal ('normal'): Clear "V" spread between Index + Middle (with Ring + Pinky folded)
 * - Naruto ('naruto'): Hands in Naruto Tiger/Ram mudra in front of chest/chin (Index+Middle up & glued together, Ring+Pinky folded)
 */
export function detectCoexistingGesture(
  worldHands: Point3D[][],
  imageHands: Point2D[][],
  faceForeheadPoint: Point2D | null,
  width: number,
  height: number
): { mode: VanishMode; score: number } | null {
  if (!imageHands || imageHands.length === 0) return null;

  // 1. Check Goku (2 fingers standing up & pressed together on the forehead)
  if (isGokuForeheadGesture(imageHands, faceForeheadPoint)) {
    return { mode: 'goku', score: 1.0 };
  }

  // 2. Check V-Sign (clear V-spread between Index and Middle)
  const vScore = classifyVictoryFromHands(worldHands, imageHands, width, height);
  if (vScore >= 0.62) {
    return { mode: 'normal', score: vScore };
  }

  // 3. Check Naruto (Tiger/Ram mudra in front of torso/chin, 2 hands or merged interlocked hands)
  if (isNarutoHandSeal(imageHands, width, height, faceForeheadPoint)) {
    return { mode: 'naruto', score: 1.0 };
  }

  return null;
}

/**
 * Exact port of `person_mask` from `Nicovic17/vanish_webcam/vanish_cam.py`:
 * 1. Bilinear interpolation (`cv2.INTER_LINEAR`) from raw MediaPipe mask `(mW, mH)` to `(width, height)`
 * 2. Separable Gaussian blur (`sigma = 2.2`) to eliminate 256x144 staircase blocks
 * 3. Contrast stretch & clip: `np.clip((resized - 0.12) / 0.68, 0.0, 1.0)`
 */
export function upsampleAndSmoothMask(
  rawMask: Float32Array,
  mW: number,
  mH: number,
  width: number,
  height: number,
  mirror: boolean,
  outMask: Float32Array,
  tempMask?: Float32Array
): void {
  const scratch =
    tempMask && tempMask.length === width * height
      ? tempMask
      : new Float32Array(width * height);

  const scaleX = mW / width;
  const scaleY = mH / height;

  // Step 1: Bilinear interpolation (cv2.INTER_LINEAR)
  for (let y = 0; y < height; y++) {
    const srcY = Math.max(0, Math.min(mH - 1, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(srcY);
    const y1 = Math.min(mH - 1, y0 + 1);
    const wy = srcY - y0;
    const row0 = y0 * mW;
    const row1 = y1 * mW;
    const dstRow = y * width;

    for (let x = 0; x < width; x++) {
      const effX = mirror ? width - 1 - x : x;
      const srcX = Math.max(0, Math.min(mW - 1, (effX + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(srcX);
      const x1 = Math.min(mW - 1, x0 + 1);
      const wx = srcX - x0;

      const top = rawMask[row0 + x0] * (1 - wx) + rawMask[row0 + x1] * wx;
      const bottom = rawMask[row1 + x0] * (1 - wx) + rawMask[row1 + x1] * wx;
      scratch[dstRow + x] = top * (1 - wy) + bottom * wy;
    }
  }

  // Step 2: Separable 7-tap Gaussian blur (sigma = 2.2) + Step 3: clip((val - 0.12) / 0.68, 0, 1)
  // Kernel weights for sigma=2.2 at offsets [-3, -2, -1, 0, 1, 2, 3]
  const k0 = 0.2146;
  const k1 = 0.1933;
  const k2 = 0.1409;
  const k3 = 0.0828;

  // Horizontal pass: scratch -> outMask
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const xm3 = Math.max(0, x - 3);
      const xm2 = Math.max(0, x - 2);
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(width - 1, x + 1);
      const xp2 = Math.min(width - 1, x + 2);
      const xp3 = Math.min(width - 1, x + 3);

      outMask[row + x] =
        scratch[row + x] * k0 +
        (scratch[row + xm1] + scratch[row + xp1]) * k1 +
        (scratch[row + xm2] + scratch[row + xp2]) * k2 +
        (scratch[row + xm3] + scratch[row + xp3]) * k3;
    }
  }

  // Vertical pass + normalization: outMask -> scratch -> outMask
  for (let y = 0; y < height; y++) {
    const ym3 = Math.max(0, y - 3) * width;
    const ym2 = Math.max(0, y - 2) * width;
    const ym1 = Math.max(0, y - 1) * width;
    const y0 = y * width;
    const yp1 = Math.min(height - 1, y + 1) * width;
    const yp2 = Math.min(height - 1, y + 2) * width;
    const yp3 = Math.min(height - 1, y + 3) * width;

    for (let x = 0; x < width; x++) {
      const blurred =
        outMask[y0 + x] * k0 +
        (outMask[ym1 + x] + outMask[yp1 + x]) * k1 +
        (outMask[ym2 + x] + outMask[yp2 + x]) * k2 +
        (outMask[ym3 + x] + outMask[yp3 + x]) * k3;

      scratch[y0 + x] = Math.max(0.0, Math.min(1.0, (blurred - 0.12) / 0.68));
    }
  }

  outMask.set(scratch);
}

/**
 * Computes a binary dilated safety mask (`Uint8Array` of 0 or 1) around any pixel
 * where `mask[i] > threshold`. Used to prevent stray hair, ears, or shoulder halos
 * from ever being written into the background buffer (`bgBuffer`).
 */
export function computeDilatedPersonMask(
  mask: Float32Array,
  width: number,
  height: number,
  radiusPx: number = 26,
  threshold: number = 0.03,
  outDilated?: Uint8Array
): Uint8Array {
  const horiz = new Uint8Array(width * height);
  const result =
    outDilated && outDilated.length === width * height
      ? outDilated
      : new Uint8Array(width * height);

  // Horizontal dilation using running window counter
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let count = 0;
    for (let x = 0; x <= Math.min(width - 1, radiusPx); x++) {
      if (mask[row + x] > threshold) count++;
    }
    for (let x = 0; x < width; x++) {
      horiz[row + x] = count > 0 ? 1 : 0;
      const addX = x + radiusPx + 1;
      if (addX < width && mask[row + addX] > threshold) count++;
      const remX = x - radiusPx;
      if (remX >= 0 && mask[row + remX] > threshold) count--;
    }
  }

  // Vertical dilation using running window counter
  for (let x = 0; x < width; x++) {
    let count = 0;
    for (let y = 0; y <= Math.min(height - 1, radiusPx); y++) {
      if (horiz[y * width + x] > 0) count++;
    }
    for (let y = 0; y < height; y++) {
      result[y * width + x] = count > 0 ? 1 : 0;
      const addY = y + radiusPx + 1;
      if (addY < height && horiz[addY * width + x] > 0) count++;
      const remY = y - radiusPx;
      if (remY >= 0 && horiz[remY * width + x] > 0) count--;
    }
  }

  return result;
}

/**
 * Fills any background pixels that have never been observed person-free (`bgCleanMap[i] === 0`)
 * by smoothly interpolating from the nearest clean left, right, and top background pixels.
 * Prevents ghost faces, ears, or demo textures from ever appearing if a user triggers
 * a vanish effect without stepping completely out of the frame during calibration.
 */
export function inpaintUncalibratedBackground(
  bgBuffer: Uint8ClampedArray,
  bgCleanMap: Uint8Array,
  width: number,
  height: number
): void {
  // Check if any clean pixels exist at all
  let hasAnyClean = false;
  for (let i = 0; i < width * height; i++) {
    if (bgCleanMap[i] === 1) {
      hasAnyClean = true;
      break;
    }
  }
  if (!hasAnyClean) return;

  // Row-by-row horizontal + vertical directional interpolation across uncalibrated spans
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let x = 0;
    while (x < width) {
      if (bgCleanMap[row + x] === 1) {
        x++;
        continue;
      }

      // Found start of an uncalibrated span [startX .. endX - 1]
      const startX = x;
      while (x < width && bgCleanMap[row + x] === 0) {
        x++;
      }
      const endX = x; // first clean pixel on the right (or width)

      const hasLeft = startX > 0 && bgCleanMap[row + startX - 1] === 1;
      const hasRight = endX < width && bgCleanMap[row + endX] === 1;

      const leftIdx = hasLeft ? (row + startX - 1) * 4 : (row + Math.min(width - 1, endX)) * 4;
      const rightIdx = hasRight ? (row + endX) * 4 : leftIdx;

      const lr = bgBuffer[leftIdx];
      const lg = bgBuffer[leftIdx + 1];
      const lb = bgBuffer[leftIdx + 2];

      const rr = bgBuffer[rightIdx];
      const rg = bgBuffer[rightIdx + 1];
      const rb = bgBuffer[rightIdx + 2];

      const spanLen = Math.max(1, endX - startX);

      for (let ix = startX; ix < endX; ix++) {
        const t = hasLeft && hasRight ? (ix - startX + 0.5) / spanLen : hasLeft ? 0.0 : 1.0;
        const horizR = lr * (1 - t) + rr * t;
        const horizG = lg * (1 - t) + rg * t;
        const horizB = lb * (1 - t) + rb * t;

        // Blend with above pixel (if y > 0) for smooth 2D wall continuity
        const p = (row + ix) * 4;
        if (y > 0) {
          const upP = ((y - 1) * width + ix) * 4;
          bgBuffer[p] = Math.round(horizR * 0.65 + bgBuffer[upP] * 0.35);
          bgBuffer[p + 1] = Math.round(horizG * 0.65 + bgBuffer[upP + 1] * 0.35);
          bgBuffer[p + 2] = Math.round(horizB * 0.65 + bgBuffer[upP + 2] * 0.35);
          bgBuffer[p + 3] = 255;
        } else {
          bgBuffer[p] = Math.round(horizR);
          bgBuffer[p + 1] = Math.round(horizG);
          bgBuffer[p + 2] = Math.round(horizB);
          bgBuffer[p + 3] = 255;
        }
      }
    }
  }
}

/**
 * Exact port of `alpha_composite` from `Nicovic17/vanish_webcam/vanish_cam.py`:
 * `person * alpha + background * (1.0 - alpha)` where `alpha = mask[i] * survival[i]`.
 * Guarantees that when `survival == 0`, `outData == bgBuffer` everywhere (0% ghost rim).
 */
export function alphaCompositeConstantSurvival(
  liveData: Uint8ClampedArray,
  bgBuffer: Uint8ClampedArray,
  outData: Uint8ClampedArray,
  mask: Float32Array,
  survival: number
): void {
  if (survival <= 0.001) {
    outData.set(bgBuffer);
    return;
  }
  const len = mask.length;
  for (let i = 0; i < len; i++) {
    const alpha = mask[i] * survival;
    const p = i * 4;
    if (alpha <= 0.002) {
      outData[p] = bgBuffer[p];
      outData[p + 1] = bgBuffer[p + 1];
      outData[p + 2] = bgBuffer[p + 2];
      outData[p + 3] = 255;
    } else {
      const inv = 1.0 - alpha;
      outData[p] = liveData[p] * alpha + bgBuffer[p] * inv;
      outData[p + 1] = liveData[p + 1] * alpha + bgBuffer[p + 1] * inv;
      outData[p + 2] = liveData[p + 2] * alpha + bgBuffer[p + 2] * inv;
      outData[p + 3] = 255;
    }
  }
}

export function buildCleanFeedUrl(baseUrl: string, effectId: string, mirror: boolean): string {
  try {
    const url = new URL(baseUrl);
    url.searchParams.set('clean', '1');
    url.searchParams.set('defaultEffect', effectId);
    if (mirror) {
      url.searchParams.set('mirror', '1');
    } else {
      url.searchParams.delete('mirror');
    }
    return url.toString();
  } catch {
    return `${baseUrl}?clean=1&defaultEffect=${encodeURIComponent(effectId)}`;
  }
}

/**
 * Renders Goku's Dragon Ball Z Instant Transmission (Shunkan Idō / Téléportation)
 * faithfully matching the 2 anime reference frames:
 * - Stage 1 (0.00..0.38, Ref 1): Vertical color streak smearing along each column + dense vertical
 *   black & white anime ink speed-lines raining across the silhouette.
 * - Stage 2 (0.14..0.86, Ref 2): Vertical background-cutout bands (stepped vertical slits of varying
 *   widths 4px..20px) that slice cleanly through the silhouette to reveal `bgBuffer` underneath,
 *   bordered by crisp 1px–2px cel-shaded black ink outlines and clusters of fine vertical black speed-lines.
 * - Stage 3 (0.86..1.00): Final vertical speed-line snap into 100% clean `bgBuffer`.
 */
export function renderGokuShunkanIdo(
  liveData: Uint8ClampedArray,
  bgBuffer: Uint8ClampedArray,
  outData: Uint8ClampedArray,
  maskBuffer: Float32Array,
  width: number,
  height: number,
  effectiveProgress: number,
  nowMs: number,
  ctx?: CanvasRenderingContext2D
): void {
  const p = Math.max(0.0, Math.min(1.0, effectiveProgress));

  // Fast exit once fully vanished
  if (p >= 0.94) {
    outData.set(bgBuffer);
    if (ctx) {
      ctx.putImageData(new ImageData(new Uint8ClampedArray(outData), width, height), 0, 0);
    }
    return;
  }

  // Find silhouette bounding box so vertical slits and stepped tips scale to the person
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY = 0;
  let foundPixels = 0;

  for (let y = 0; y < height; y += 3) {
    const row = y * width;
    for (let x = 0; x < width; x += 3) {
      if (maskBuffer[row + x] > 0.28) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        foundPixels++;
      }
    }
  }

  if (foundPixels < 15 || maxX <= minX || maxY <= minY) {
    minX = Math.floor(width * 0.25);
    maxX = Math.floor(width * 0.75);
    minY = Math.floor(height * 0.12);
    maxY = Math.floor(height * 0.95);
  }

  const silHeight = Math.max(40, maxY - minY);
  // 20fps anime cell-animation step so speed lines & vertical slits snap like hand-drawn DBZ frames
  const animeStep = Math.floor(nowMs / 48);

  // Precompute per-column vertical strip slicing state (Ref 2) + vertical streak smear (Ref 1)
  const colCutoutTop = new Int16Array(width);
  const colCutoutBottom = new Int16Array(width);
  const colIsCutout = new Uint8Array(width);
  const colShiftY = new Int16Array(width);
  const colStreakDy = new Int16Array(width);

  // Variable strip widths matching the hand-drawn vertical bands in Ref 2
  const stripWidths = [15, 9, 19, 7, 13, 17, 8, 22, 10, 14, 6, 18, 11, 16];
  let curX = 0;
  let stripIdx = 0;

  while (curX < width) {
    const baseW = stripWidths[stripIdx % stripWidths.length];
    const stripW = Math.min(width - curX, baseW);

    // Deterministic hash for this vertical strip
    const h1 = Math.abs(Math.sin(stripIdx * 12.9898 + 4.1414) * 43758.5453) % 1;
    const h2 = Math.abs(Math.sin(stripIdx * 78.233 + 1.618) * 19341.123) % 1;
    const h3 = Math.abs(Math.sin((stripIdx + animeStep * 0.35) * 31.71) * 27183.81) % 1;

    // Threshold in [0.12 .. 0.64] when this vertical band starts slicing open to reveal bgBuffer
    const openAt = 0.12 + h1 * 0.52;
    // Vertical cell-slide offset (-18px .. +18px) for visible strips
    const slideDir = stripIdx % 2 === 0 ? -1 : 1;
    const stripSlide = Math.round(
      slideDir * (3 + h2 * 15) * Math.pow(Math.max(0, (p - 0.08) / 0.8), 1.2)
    );

    // How wide the background cutout slit is inside this strip at progress `p`
    const openRatio =
      p <= openAt ? 0.0 : Math.min(1.0, (p - openAt) / Math.max(0.14, 0.84 - openAt));
    // Widens from center of strip outward; above p=0.78 all strips converge to 100% open
    const globalCollapse = p > 0.72 ? (p - 0.72) / 0.16 : 0.0;
    const effectiveOpenRatio = Math.min(1.0, openRatio * 0.92 + globalCollapse);
    const slitHalfW = (stripW * 0.52) * effectiveOpenRatio;
    const stripCenter = curX + stripW * 0.5;

    for (let dx = 0; dx < stripW; dx++) {
      const x = curX + dx;
      colShiftY[x] = stripSlide;

      // High-frequency vertical streak offset (Ref 1: vertical motion blur along columns)
      const colHash = Math.sin(x * 1.93 + animeStep * 2.7);
      colStreakDy[x] = Math.round(colHash * (7 + p * 18));

      const distFromStripCenter = Math.abs(x + 0.5 - stripCenter);
      if (effectiveOpenRatio > 0.04 && distFromStripCenter <= slitHalfW) {
        colIsCutout[x] = 1;

        // Stepped/crenelated top and bottom slit tips inside the silhouette (as seen in Ref 2!)
        // As `effectiveOpenRatio` grows past 0.45, the slit cuts all the way from top (0) to bottom (height)
        const subColStep = dx < stripW * 0.35 ? 0.14 : dx > stripW * 0.65 ? 0.09 : 0.0;
        const vSpanRatio = Math.min(1.0, effectiveOpenRatio * 1.45 - subColStep);
        if (vSpanRatio >= 0.88) {
          colCutoutTop[x] = 0;
          colCutoutBottom[x] = height;
        } else {
          const anchorY = minY + silHeight * (0.25 + 0.5 * h3);
          const halfSpan = silHeight * (0.28 + 0.65 * vSpanRatio);
          colCutoutTop[x] = Math.max(0, Math.floor(anchorY - halfSpan));
          colCutoutBottom[x] = Math.min(height, Math.ceil(anchorY + halfSpan));
        }
      } else {
        colIsCutout[x] = 0;
      }
    }

    curX += stripW;
    stripIdx++;
  }

  // Stage 1 vertical smear weight (Ref 1: strong in 0.0..0.45, tapers as crisp Ref 2 strips take over)
  const smearStrength =
    p < 0.42 ? Math.min(0.62, (p / 0.15) * 0.55) : Math.max(0.18, 0.55 - (p - 0.42) * 0.8);

  // Fast zero-cost initialization: set entire frame to background buffer via typed array memcpy
  outData.set(bgBuffer);

  // Precompute 1D ink border column flags for ultra-fast single-instruction lookups in inner loop
  const colLeftBorder = new Uint8Array(width);
  const colRightBorder = new Uint8Array(width);
  if (p >= 0.14) {
    for (let x = 0; x < width; x++) {
      if (colIsCutout[x] === 1) {
        if (x > 0 && colIsCutout[x - 1] === 0) colLeftBorder[x] = 1;
        if (x < width - 1 && colIsCutout[x + 1] === 0) colRightBorder[x] = 1;
      }
    }
  }

  // Pass 1: Pixel-level vertical strip slicing, cel-shaded black ink borders, and vertical streak smearing
  // Confined to the silhouette ROI bounding box (300% faster)
  const yStart = Math.max(0, minY - 4);
  const yEnd = Math.min(height, maxY + 4);
  const xStart = Math.max(0, minX - 4);
  const xEnd = Math.min(width, maxX + 4);

  for (let y = yStart; y < yEnd; y++) {
    const row = y * width;
    for (let x = xStart; x < xEnd; x++) {
      const i = row + x;
      const baseMask = maskBuffer[i];
      if (baseMask <= 0.02) continue; // Already background

      const pIdx = i * 4;

      // Check if `(x, y)` is inside an open vertical background-cutout slit (Ref 2)
      const inCutoutSlit =
        colIsCutout[x] === 1 && y >= colCutoutTop[x] && y <= colCutoutBottom[x];

      if (inCutoutSlit) {
        continue; // Already background
      }

      // Check if `(x, y)` is on the 1px–2px cel-shaded black ink border next to a vertical cutout slit (Ref 2)
      if (baseMask > 0.36 && p >= 0.14) {
        const leftCut =
          (x > 1 && colIsCutout[x - 1] === 1 && y >= colCutoutTop[x - 1] && y <= colCutoutBottom[x - 1]) ||
          (x > 2 && colIsCutout[x - 2] === 1 && y >= colCutoutTop[x - 2] && y <= colCutoutBottom[x - 2]);
        const rightCut =
          (x < width - 2 && colIsCutout[x + 1] === 1 && y >= colCutoutTop[x + 1] && y <= colCutoutBottom[x + 1]) ||
          (x < width - 3 && colIsCutout[x + 2] === 1 && y >= colCutoutTop[x + 2] && y <= colCutoutBottom[x + 2]);
        const topBottomSteppedEdge =
          colIsCutout[x] === 1 &&
          (Math.abs(y - colCutoutTop[x]) <= 2 || Math.abs(y - colCutoutBottom[x]) <= 2);

        if (leftCut || rightCut || topBottomSteppedEdge) {
          outData[pIdx] = 14;
          outData[pIdx + 1] = 18;
          outData[pIdx + 2] = 26;
          outData[pIdx + 3] = 255;
          continue;
        }
      }

      // Visible vertical strip of the person: sample vertically shifted + vertically streaked pixel
      const sy = Math.max(0, Math.min(height - 1, y + colShiftY[x]));
      const srcIdx = sy * width + x;
      const shiftedMask = maskBuffer[srcIdx];
      const alpha = Math.min(1.0, Math.max(baseMask * 0.35, shiftedMask));

      if (alpha <= 0.02) {
        continue; // Already background
      }

      // Vertical streak motion blur along column `x` (Ref 1 vertical color smearing)
      const dy = colStreakDy[x];
      const syUp = Math.max(0, Math.min(height - 1, sy - dy));
      const syDown = Math.max(0, Math.min(height - 1, sy + dy));
      const p0 = srcIdx * 4;
      const pUp = (syUp * width + x) * 4;
      const pDown = (syDown * width + x) * 4;

      const wCenter = 1.0 - smearStrength;
      const wSide = smearStrength * 0.5;

      const rPerson = liveData[p0] * wCenter + (liveData[pUp] + liveData[pDown]) * wSide;
      const gPerson =
        liveData[p0 + 1] * wCenter + (liveData[pUp + 1] + liveData[pDown + 1]) * wSide;
      const bPerson =
        liveData[p0 + 2] * wCenter + (liveData[pUp + 2] + liveData[pDown + 2]) * wSide;

      const inv = 1.0 - alpha;
      outData[pIdx] = rPerson * alpha + bgBuffer[pIdx] * inv;
      outData[pIdx + 1] = gPerson * alpha + bgBuffer[pIdx + 1] * inv;
      outData[pIdx + 2] = bPerson * alpha + bgBuffer[pIdx + 2] * inv;
      outData[pIdx + 3] = 255;
    }
  }

  // Pass 2: Etch high-density vertical anime speed-lines (black ink + white/cream streaks) directly into `outData`
  // so both pixel buffer assertions and canvas output contain the iconic vertical DBZ hatching (Ref 1 & Ref 2)
  const rng = createSeededRng(4099 + animeStep * 131);
  const silWidth = Math.max(40, maxX - minX);
  // In Stage 1 (Ref 1), more white+black vertical streaks across the whole silhouette;
  // In Stage 2 (Ref 2), sharp black vertical ink speed-lines clustered inside the remaining strips
  const numSpeedLines = p < 0.4 ? 135 : p < 0.8 ? 95 : 42;

  for (let l = 0; l < numSpeedLines; l++) {
    const lx = Math.floor(minX + rng() * silWidth);
    if (lx < 1 || lx >= width - 1) continue;

    const lineLen = Math.floor(silHeight * (0.12 + rng() * 0.42));
    const lyStart = Math.floor(minY - silHeight * 0.04 + rng() * (silHeight * 0.88));
    const lyEnd = Math.min(height - 1, lyStart + lineLen);
    const midY = Math.max(0, Math.min(height - 1, (lyStart + lyEnd) >> 1));

    // Speed lines only appear over the silhouette (and not inside wide-open background slits)
    if (maskBuffer[midY * width + lx] < 0.22) continue;
    if (p >= 0.32 && colIsCutout[lx] === 1 && l % 3 !== 0) continue;

    // Ref 1 has alternating black ink & bright white/cream vertical streaks; Ref 2 is predominantly crisp black ink lines
    const isWhiteStreak = p < 0.42 ? l % 3 === 0 : l % 7 === 0;
    const isThick = l % 5 === 0;
    const lineAlpha = p > 0.8 ? Math.max(0, (0.94 - p) / 0.14) : 0.88;

    const targetR = isWhiteStreak ? 250 : 12;
    const targetG = isWhiteStreak ? 248 : 16;
    const targetB = isWhiteStreak ? 238 : 24;

    for (let y = Math.max(0, lyStart); y <= lyEnd; y++) {
      const m = maskBuffer[y * width + lx];
      if (m < 0.12) continue;
      // Taper alpha at top and bottom tips of each vertical speed line
      const normY = (y - lyStart) / Math.max(1, lyEnd - lyStart);
      const tipFade = Math.sin(normY * Math.PI);
      const a = lineAlpha * tipFade;
      const invA = 1.0 - a;

      const idx = (y * width + lx) * 4;
      outData[idx] = outData[idx] * invA + targetR * a;
      outData[idx + 1] = outData[idx + 1] * invA + targetG * a;
      outData[idx + 2] = outData[idx + 2] * invA + targetB * a;

      if (isThick && lx + 1 < width) {
        const idx2 = (y * width + lx + 1) * 4;
        const a2 = a * 0.75;
        const invA2 = 1.0 - a2;
        outData[idx2] = outData[idx2] * invA2 + targetR * a2;
        outData[idx2 + 1] = outData[idx2 + 1] * invA2 + targetG * a2;
        outData[idx2 + 2] = outData[idx2 + 2] * invA2 + targetB * a2;
      }
    }
  }

  if (ctx) {
    ctx.putImageData(new ImageData(new Uint8ClampedArray(outData), width, height), 0, 0);
  }
}
