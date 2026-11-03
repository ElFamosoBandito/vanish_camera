import test from 'node:test';
import assert from 'node:assert/strict';
import {
  makeDissolveFields,
  substitutionAnchor,
  fallingLogAnchor,
  landedLogAnchor,
  victoryLandmarkScore,
  joinedHandsFromLandmarks,
  isGokuForeheadGesture,
  detectCoexistingGesture,
  upsampleAndSmoothMask,
  computeDilatedPersonMask,
  inpaintUncalibratedBackground,
  alphaCompositeConstantSurvival,
  computeMedianBackground,
  classifyVictoryFromHands,
  isNarutoHandSeal,
  renderGokuShunkanIdo,
  type Point3D,
  type Point2D,
} from './vfxEngine';

function createHandPoints(extended: [boolean, boolean, boolean, boolean] = [true, true, false, false]): Point3D[] {
  const points: Point3D[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  const bases = [5, 9, 13, 17];
  const xs = [-0.45, 0.0, 0.4, 0.75];
  const spreads = [-0.35, 0.25, 0.0, 0.0];

  for (let i = 0; i < 4; i++) {
    const base = bases[i];
    const x = xs[i];
    const opened = extended[i];
    const spread = spreads[i];
    if (opened) {
      points[base] = { x, y: 1.0, z: 0.0 };
      points[base + 1] = { x: x + spread * 0.45, y: 1.7, z: 0.0 };
      points[base + 2] = { x: x + spread * 0.75, y: 2.15, z: 0.0 };
      points[base + 3] = { x: x + spread, y: 2.5, z: 0.0 };
    } else {
      points[base] = { x, y: 1.0, z: 0.0 };
      points[base + 1] = { x, y: 1.5, z: 0.2 };
      points[base + 2] = { x, y: 1.3, z: 0.5 };
      points[base + 3] = { x, y: 0.95, z: 0.3 };
    }
  }
  return points;
}

test('victoryLandmarkScore detects V-sign across rotations and rejects other gestures', () => {
  const victoryHand = createHandPoints([true, true, false, false]);
  assert.ok(victoryLandmarkScore(victoryHand) > 0.62);

  // Fist, Open Palm, Single Index, Three Fingers should return 0
  assert.equal(victoryLandmarkScore(createHandPoints([false, false, false, false])), 0);
  assert.equal(victoryLandmarkScore(createHandPoints([true, true, true, true])), 0);
  assert.equal(victoryLandmarkScore(createHandPoints([true, false, false, false])), 0);
  assert.equal(victoryLandmarkScore(createHandPoints([true, true, true, false])), 0);
});

function makeNarutoMudraHand(centerX: number, openAllFingers = false): Point2D[] {
  const pts: Point2D[] = Array.from({ length: 21 }, () => ({ x: centerX, y: 0.6 }));
  // Wrist
  pts[0] = { x: centerX, y: 0.72 };
  // MCP joints
  pts[5] = { x: centerX - 0.03, y: 0.58 };
  pts[9] = { x: centerX - 0.005, y: 0.57 };
  pts[13] = { x: centerX + 0.02, y: 0.59 };
  pts[17] = { x: centerX + 0.045, y: 0.61 };

  // Index (5->6->7->8) & Middle (9->10->11->12) straight up and pressed tightly together
  pts[6] = { x: centerX - 0.012, y: 0.49 };
  pts[7] = { x: centerX - 0.008, y: 0.42 };
  pts[8] = { x: centerX - 0.006, y: 0.36 };

  pts[10] = { x: centerX + 0.006, y: 0.49 };
  pts[11] = { x: centerX + 0.005, y: 0.42 };
  pts[12] = { x: centerX + 0.004, y: 0.36 };

  if (openAllFingers) {
    // Open palm: Ring & Pinky also extended straight up
    pts[14] = { x: centerX + 0.022, y: 0.49 };
    pts[15] = { x: centerX + 0.024, y: 0.43 };
    pts[16] = { x: centerX + 0.025, y: 0.37 };
    pts[18] = { x: centerX + 0.045, y: 0.51 };
    pts[19] = { x: centerX + 0.048, y: 0.45 };
    pts[20] = { x: centerX + 0.05, y: 0.4 };
  } else {
    // Naruto Mudra: Ring (16) & Pinky (20) folded/interlocked down in palm
    pts[14] = { x: centerX + 0.02, y: 0.56 };
    pts[15] = { x: centerX + 0.02, y: 0.61 };
    pts[16] = { x: centerX + 0.018, y: 0.64 };
    pts[18] = { x: centerX + 0.042, y: 0.58 };
    pts[19] = { x: centerX + 0.04, y: 0.63 };
    pts[20] = { x: centerX + 0.038, y: 0.65 };
  }

  return pts;
}

test('joinedHandsFromLandmarks detects two nearby hands in Naruto mudra and rejects open palms or single hand', () => {
  // Two hands joined in strict Naruto Mudra (Index+Middle straight up & glued, Ring+Pinky folded)
  assert.equal(
    joinedHandsFromLandmarks([makeNarutoMudraHand(0.48), makeNarutoMudraHand(0.52)], 640, 480),
    true
  );
  // Single hand must NEVER trigger Naruto seal
  assert.equal(joinedHandsFromLandmarks([makeNarutoMudraHand(0.5)], 640, 480), false);
  // Two open palms (Ring & Pinky not folded) must be rejected
  assert.equal(
    joinedHandsFromLandmarks(
      [makeNarutoMudraHand(0.48, true), makeNarutoMudraHand(0.52, true)],
      640,
      480
    ),
    false
  );
  // Hands far apart must be rejected
  assert.equal(
    joinedHandsFromLandmarks([makeNarutoMudraHand(0.22), makeNarutoMudraHand(0.82)], 640, 480),
    false
  );
});

test('isGokuForeheadGesture detects closed hand with index and middle standing up on the forehead', () => {
  const hand: Point2D[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.36 }));
  hand[0] = { x: 0.5, y: 0.44 };
  hand[5] = { x: 0.485, y: 0.34 };
  hand[6] = { x: 0.49, y: 0.29 };
  hand[8] = { x: 0.494, y: 0.24 };
  hand[9] = { x: 0.508, y: 0.34 };
  hand[10] = { x: 0.506, y: 0.29 };
  hand[12] = { x: 0.506, y: 0.24 };
  hand[16] = { x: 0.525, y: 0.36 };
  hand[20] = { x: 0.54, y: 0.38 };
  const forehead = { x: 0.5, y: 0.24 };

  assert.equal(isGokuForeheadGesture([hand], forehead), true);
  assert.equal(isGokuForeheadGesture([hand], { x: 0.5, y: 0.75 }), false);
});

test('substitutionAnchor and fallingLogAnchor track person mask and simulate gravity', () => {
  const width = 240;
  const height = 160;
  const mask = new Float32Array(width * height);
  for (let y = 20; y < 130; y++) {
    for (let x = 140; x < 210; x++) {
      mask[y * width + x] = 1.0;
    }
  }

  const [cx, cy, size] = substitutionAnchor(mask, width, height);
  assert.ok(cx > 140);
  assert.ok(size > 35);
  assert.ok(cy > 20 && cy < 130);

  const anchor: [number, number, number] = [120, 80, 90];
  const y0 = fallingLogAnchor(anchor, 240, 0.1)[1];
  const y1 = fallingLogAnchor(anchor, 240, 0.3)[1];
  const y2 = fallingLogAnchor(anchor, 240, 0.5)[1];
  const floorY = landedLogAnchor(anchor, 240)[1];
  assert.equal(y0, 80);
  assert.ok(y1 > y0);
  assert.ok(y2 > y1);
  assert.ok(floorY >= y2);
});

test('detectCoexistingGesture distinguishes Goku forehead, V-sign Fade Out, and Naruto seal (2-hand or merged 1-hand Tiger mudra)', () => {
  const width = 960;
  const height = 540;
  const forehead = { x: 0.5, y: 0.2 };

  // 1. Goku: closed hand + 2 fingers standing up & pressed together on the forehead -> 'goku'
  const gokuHand: Point2D[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.35 }));
  gokuHand[0] = { x: 0.5, y: 0.42 };
  gokuHand[5] = { x: 0.488, y: 0.32 };
  gokuHand[6] = { x: 0.492, y: 0.27 };
  gokuHand[8] = { x: 0.495, y: 0.22 };
  gokuHand[9] = { x: 0.51, y: 0.32 };
  gokuHand[10] = { x: 0.506, y: 0.27 };
  gokuHand[12] = { x: 0.505, y: 0.22 };
  gokuHand[16] = { x: 0.53, y: 0.33 };
  gokuHand[20] = { x: 0.55, y: 0.35 };
  const gokuMatch = detectCoexistingGesture([], [gokuHand], forehead, width, height);
  assert.ok(gokuMatch);
  assert.equal(gokuMatch.mode, 'goku');

  // 2. Naruto (2 hands detected): Two hands joined in front of chest in Tiger/Ram mudra -> 'naruto'
  const narutoHands = [makeNarutoMudraHand(0.485), makeNarutoMudraHand(0.515)];
  const narutoMatch2H = detectCoexistingGesture([], narutoHands, forehead, width, height);
  assert.ok(narutoMatch2H);
  assert.equal(narutoMatch2H.mode, 'naruto');

  // 3. Naruto (1 merged hand when MediaPipe fuses interlocked Tiger-seal hands in chest zone) -> 'naruto'
  const mergedTigerHand = makeNarutoMudraHand(0.5);
  const narutoMatch1H = detectCoexistingGesture([], [mergedTigerHand], forehead, width, height);
  assert.ok(narutoMatch1H);
  assert.equal(narutoMatch1H.mode, 'naruto');

  // 4. Open palm in chest zone must NOT trigger any vanish
  const openPalmChest = makeNarutoMudraHand(0.5, true);
  assert.equal(detectCoexistingGesture([], [openPalmChest], forehead, width, height), null);

  // 5. Normal Fade Out: Spread V-sign -> returns 'normal' (and never 'naruto' or 'goku')
  const v3d = createHandPoints([true, true, false, false]);
  const v2d: Point2D[] = v3d.map((p) => ({ x: p.x, y: p.y }));
  const normalMatch = detectCoexistingGesture([v3d], [v2d], forehead, width, height);
  assert.ok(normalMatch);
  assert.equal(normalMatch.mode, 'normal');

  const { field, sparkleNoise } = makeDissolveFields(64, 48);
  assert.equal(field.length, 64 * 48);
  assert.equal(sparkleNoise.length, 64 * 48);
});

test('upsampleAndSmoothMask, computeDilatedPersonMask, inpaintUncalibratedBackground, and alphaComposite eliminate ghost rims', () => {
  const mW = 16;
  const mH = 16;
  const raw = new Float32Array(mW * mH);
  for (let y = 4; y < 12; y++) {
    for (let x = 4; x < 12; x++) {
      raw[y * mW + x] = 0.95;
    }
  }
  const W = 32;
  const H = 32;
  const outMask = new Float32Array(W * H);
  upsampleAndSmoothMask(raw, mW, mH, W, H, false, outMask);

  // Center should be 1.0, corner should be 0.0, boundary should be smoothly feathered
  assert.equal(outMask[16 * W + 16], 1.0);
  assert.equal(outMask[0], 0.0);

  // Dilation by 4px should mark pixels outside the raw boundary as protected (1)
  const dilated = computeDilatedPersonMask(outMask, W, H, 4, 0.05);
  assert.equal(dilated[16 * W + 16], 1);
  assert.equal(dilated[16 * W + 6], 1); // outside raw center but within 4px dilation

  // Background inpainting should fill uncalibrated center span from left/right clean pixels
  const bg = new Uint8ClampedArray(W * H * 4);
  const cleanMap = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      if (x < 8 || x > 24) {
        cleanMap[idx] = 1;
        bg[idx * 4] = 100;
        bg[idx * 4 + 1] = 150;
        bg[idx * 4 + 2] = 200;
        bg[idx * 4 + 3] = 255;
      }
    }
  }
  inpaintUncalibratedBackground(bg, cleanMap, W, H);
  assert.equal(bg[(16 * W + 16) * 4], 100);
  assert.equal(bg[(16 * W + 16) * 4 + 1], 150);
  assert.equal(bg[(16 * W + 16) * 4 + 2], 200);

  // Alpha composite at survival=0 must leave 0% of liveData even where mask is partial (0.4)
  const live = new Uint8ClampedArray(W * H * 4).fill(255);
  const out = new Uint8ClampedArray(W * H * 4);
  outMask[5 * W + 5] = 0.4; // simulate soft ear/hair edge
  alphaCompositeConstantSurvival(live, bg, out, outMask, 0.0);
  assert.equal(out[(5 * W + 5) * 4], bg[(5 * W + 5) * 4]);

  // computeMedianBackground rejects transient outlier frames (matching np.median(stack[-18:], axis=0))
  const f1 = new Uint8ClampedArray([10, 20, 30, 255]);
  const f2 = new Uint8ClampedArray([12, 22, 32, 255]);
  const f3Outlier = new Uint8ClampedArray([250, 250, 250, 255]);
  const medOut = new Uint8ClampedArray(4);
  computeMedianBackground([f1, f2, f3Outlier], medOut);
  assert.equal(medOut[0], 12);
  assert.equal(medOut[1], 22);
  assert.equal(medOut[2], 32);

  // classifyVictoryFromHands detects V-sign from 2D landmarks projected to (W, H, 0)
  const v3d = createHandPoints([true, true, false, false]);
  const v2d: Point2D[] = v3d.map((p) => ({ x: p.x, y: p.y }));
  assert.ok(classifyVictoryFromHands([], [v2d], 960, 540) > 0.62);

  // isNarutoHandSeal accepts both 2-hand and merged 1-hand Tiger mudra in the chest zone, and rejects open palms
  const twoHands = [makeNarutoMudraHand(0.485), makeNarutoMudraHand(0.515)];
  assert.equal(isNarutoHandSeal(twoHands, 960, 540), true);
  assert.equal(isNarutoHandSeal([makeNarutoMudraHand(0.5)], 960, 540), true);
  assert.equal(isNarutoHandSeal([makeNarutoMudraHand(0.5, true)], 960, 540), false);
});

test('renderGokuShunkanIdo slices silhouette into vertical background-cutout bands with black ink borders and restores clean background at completion', () => {
  const W = 120;
  const H = 90;
  const live = new Uint8ClampedArray(W * H * 4);
  const bg = new Uint8ClampedArray(W * H * 4);
  const out = new Uint8ClampedArray(W * H * 4);
  const mask = new Float32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const p = i * 4;
      bg[p] = 40;
      bg[p + 1] = 180;
      bg[p + 2] = 220;
      bg[p + 3] = 255;

      live[p] = 240;
      live[p + 1] = 140;
      live[p + 2] = 90;
      live[p + 3] = 255;

      if (x >= 25 && x <= 95 && y >= 12 && y <= 80) {
        mask[i] = 1.0;
      }
    }
  }

  // At p = 0.48 (mid-teleportation, matching Reference Image 2):
  // Inside the silhouette [x=25..95, y=45], some columns must be sliced open to pure `bg` (40,180,220),
  // some columns must remain visible person strips, and cel-shaded ink borders (14,18,26) must appear between them.
  renderGokuShunkanIdo(live, bg, out, mask, W, H, 0.48, 1000);

  let bgCutoutCols = 0;
  let personCols = 0;
  let inkBorderCols = 0;
  const midRow = 45 * W;

  for (let x = 30; x <= 90; x++) {
    const p = (midRow + x) * 4;
    const r = out[p];
    const g = out[p + 1];
    const b = out[p + 2];
    if (r === 40 && g === 180 && b === 220) {
      bgCutoutCols++;
    } else if (r === 14 && g === 18 && b === 26) {
      inkBorderCols++;
    } else if (r > 100) {
      personCols++;
    }
  }

  assert.ok(bgCutoutCols > 5, 'Should slice open multiple vertical background-cutout columns');
  assert.ok(personCols > 5, 'Should retain vertical strips of the person silhouette');
  assert.ok(inkBorderCols > 2, 'Should draw cel-shaded black ink borders along vertical cut edges');

  // At p = 0.96, 100% of the frame must be restored to pure calibrated background
  renderGokuShunkanIdo(live, bg, out, mask, W, H, 0.96, 1000);
  for (let x = 25; x <= 95; x++) {
    const p = (midRow + x) * 4;
    assert.equal(out[p], 40);
    assert.equal(out[p + 1], 180);
    assert.equal(out[p + 2], 220);
  }
});
