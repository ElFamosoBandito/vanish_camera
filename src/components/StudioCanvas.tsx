import React, { useEffect, useRef } from 'react';
import {
  FilesetResolver,
  ImageSegmenter,
  HandLandmarker,
  FaceLandmarker,
} from '@mediapipe/tasks-vision';
import {
  type AppPhase,
  type VanishMode,
  CALIBRATION_SECONDS,
  BACKGROUND_SAMPLE_SECONDS,
  NARUTO_EFFECT_SECONDS,
  createSeededRng,
  easeInOutCubic,
  substitutionAnchor,
  fallingLogAnchor,
  landedLogAnchor,
  detectCoexistingGesture,
  upsampleAndSmoothMask,
  computeMedianBackground,
  alphaCompositeConstantSurvival,
  renderGokuShunkanIdo,
} from '../utils/vfxEngine';

interface StudioCanvasProps {
  mode: VanishMode;
  phase: AppPhase;
  progress: number; // 0..1
  calibrationRemainingSec: number;
  useRealWebcam: boolean;
  mirrorVideo: boolean;
  gokuHoldProgress: number; // 0..1
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  bgCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  onGestureDetected: (detectedMode: VanishMode, detail: string) => void;
  onGokuHoldProgress: (holdRatio: number) => void;
  onMediaPipeStatusChange: (status: string) => void;
  onCalibrationComplete: () => void;
  onCalibrationTick: (remainingSec: number) => void;
}

// Exact default resolution from Nicovic17/vanish_webcam/vanish_cam.py (--width 960 --height 540)
const W = 960;
const H = 540;

export const StudioCanvas: React.FC<StudioCanvasProps> = ({
  mode,
  phase,
  progress,
  calibrationRemainingSec,
  useRealWebcam,
  mirrorVideo,
  videoRef,
  canvasRef,
  maskCanvasRef,
  bgCanvasRef,
  onGestureDetected,
  onGokuHoldProgress,
  onMediaPipeStatusChange,
  onCalibrationComplete,
  onCalibrationTick,
}) => {
  const propsRef = useRef({
    mode,
    phase,
    progress,
    calibrationRemainingSec,
    useRealWebcam,
    mirrorVideo,
    onGestureDetected,
    onGokuHoldProgress,
    onCalibrationComplete,
    onCalibrationTick,
  });

  useEffect(() => {
    propsRef.current = {
      mode,
      phase,
      progress,
      calibrationRemainingSec,
      useRealWebcam,
      mirrorVideo,
      onGestureDetected,
      onGokuHoldProgress,
      onCalibrationComplete,
      onCalibrationTick,
    };
  }, [
    mode,
    phase,
    progress,
    calibrationRemainingSec,
    useRealWebcam,
    mirrorVideo,
    onGestureDetected,
    onGokuHoldProgress,
    onCalibrationComplete,
    onCalibrationTick,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Offscreen work canvases for pixel-level segmentation & compositing
    const liveWorkCanvas = document.createElement('canvas');
    liveWorkCanvas.width = W;
    liveWorkCanvas.height = H;
    const liveCtx = liveWorkCanvas.getContext('2d', { willReadFrequently: true })!;

    const personPhotoCanvas = document.createElement('canvas');
    personPhotoCanvas.width = W;
    personPhotoCanvas.height = H;
    const personPhotoCtx = personPhotoCanvas.getContext('2d', { willReadFrequently: true })!;

    // Buffers matching vanish_cam.py:
    // `bgBuffer` is ONLY captured during the last 0.55s of the 4.0s calibration via np.median(background_samples[-18:])
    // and is NEVER mutated while the user is in the frame!
    const demoRoomBuffer = new Uint8ClampedArray(W * H * 4);
    const bgBuffer = new Uint8ClampedArray(W * H * 4);
    let backgroundSamples: Uint8ClampedArray[] = [];
    let bgInitialized = false;
    let lastWebcamState = false;
    let calibrationStartedMs = performance.now();
    let lastReportedRemaining = -1;

    const maskBuffer = new Float32Array(W * H);
    const maskScratch = new Float32Array(W * H);
    const demoPhotoMask = new Float32Array(W * H);
    const edgeGlowBuffer = new Float32Array(W * H);
    const outImageData = ctx.createImageData(W, H);
    const outData = outImageData.data;

    // Load real photographic reference assets for demo preview (when physical webcam is not active)
    const roomImg = new Image();
    roomImg.src = '/assets/chairs-image-seg.jpg';
    let roomLoaded = false;

    const personImg = new Image();
    personImg.src = '/assets/business-person.png';
    let personLoaded = false;

    const logSprite = new Image();
    logSprite.src = '/assets/substitution_log.png';
    const logTrimmedCanvas = document.createElement('canvas');
    let logLoaded = false;

    roomImg.onload = () => {
      roomLoaded = true;
      liveCtx.drawImage(roomImg, 0, 0, W, H);
      const roomData = liveCtx.getImageData(0, 0, W, H).data;
      demoRoomBuffer.set(roomData);
      if (!propsRef.current.useRealWebcam) {
        bgBuffer.set(demoRoomBuffer);
        bgInitialized = true;
      }
    };

    personImg.onload = () => {
      personLoaded = true;
      personPhotoCtx.clearRect(0, 0, W, H);
      const pH = Math.round(H * 0.88);
      const pW = Math.round((personImg.width / personImg.height) * pH);
      const px = Math.round((W - pW) * 0.5);
      const py = H - pH;
      personPhotoCtx.drawImage(personImg, px, py, pW, pH);
      const pImgData = personPhotoCtx.getImageData(0, 0, W, H);
      const pData = pImgData.data;

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          const r = pData[i * 4];
          const g = pData[i * 4 + 1];
          const b = pData[i * 4 + 2];
          const a = pData[i * 4 + 3];

          const minChannel = Math.min(r, g, b);
          const maxChannel = Math.max(r, g, b);
          const saturation = maxChannel - minChannel;

          const isInnerBlouse =
            x >= W * 0.455 &&
            x <= W * 0.535 &&
            y >= H * 0.39 &&
            y <= H * 0.98;

          if (isInnerBlouse && a > 200) {
            demoPhotoMask[i] = 1.0;
            pData[i * 4 + 3] = 255;
            continue;
          }

          if (a < 30 || (minChannel > 224 && saturation < 16)) {
            demoPhotoMask[i] = 0.0;
            pData[i * 4 + 3] = 0;
          } else if (minChannel > 200 && saturation < 18) {
            const alphaVal = Math.max(0, Math.min(1, (224 - minChannel) / 24));
            demoPhotoMask[i] = alphaVal;
            pData[i * 4] = Math.round(r * alphaVal);
            pData[i * 4 + 1] = Math.round(g * alphaVal);
            pData[i * 4 + 2] = Math.round(b * alphaVal);
            pData[i * 4 + 3] = Math.round(alphaVal * 255);
          } else {
            demoPhotoMask[i] = 1.0;
            pData[i * 4 + 3] = 255;
          }
        }
      }

      personPhotoCtx.putImageData(pImgData, 0, 0);
      maskBuffer.set(demoPhotoMask);
    };

    logSprite.onload = () => {
      const sw = logSprite.naturalWidth || logSprite.width;
      const sh = logSprite.naturalHeight || logSprite.height;
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = sw;
      tmpCanvas.height = sh;
      const tmpCtx = tmpCanvas.getContext('2d')!;
      tmpCtx.drawImage(logSprite, 0, 0);
      const imgData = tmpCtx.getImageData(0, 0, sw, sh);
      const data = imgData.data;

      let minX = sw;
      let minY = sh;
      let maxX = 0;
      let maxY = 0;

      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const idx = (y * sw + x) * 4;
          const a = data[idx + 3];
          if (a > 20) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (maxX > minX && maxY > minY) {
        const cw = maxX - minX + 1;
        const ch = maxY - minY + 1;
        logTrimmedCanvas.width = cw;
        logTrimmedCanvas.height = ch;
        const trimCtx = logTrimmedCanvas.getContext('2d')!;
        trimCtx.drawImage(tmpCanvas, minX, minY, cw, ch, 0, 0, cw, ch);
      } else {
        logTrimmedCanvas.width = sw;
        logTrimmedCanvas.height = sh;
        const trimCtx = logTrimmedCanvas.getContext('2d')!;
        trimCtx.drawImage(logSprite, 0, 0);
      }
      logLoaded = true;
    };

    // Initialize MediaPipe Tasks Vision (ImageSegmenter, HandLandmarker, FaceLandmarker)
    let segmenter: ImageSegmenter | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let faceLandmarker: FaceLandmarker | null = null;
    let disposed = false;

    const initMediaPipe = async () => {
      try {
        onMediaPipeStatusChange('Chargement MediaPipe WASM & Selfie Segmenter...');
        const vision = await FilesetResolver.forVisionTasks('/wasm');
        if (disposed) return;

        segmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/selfie_segmenter_landscape.tflite',
          },
          runningMode: 'VIDEO',
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        });

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/hand_landmarker.task',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.35,
          minHandPresenceConfidence: 0.35,
          minTrackingConfidence: 0.45,
        });

        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/face_landmarker.task',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
        });

        if (!disposed) {
          onMediaPipeStatusChange('MediaPipe SelfieSegmenter + Hand/Face Landmarker actifs');
        }
      } catch {
        if (!disposed) {
          onMediaPipeStatusChange('Segmentation locale active');
        }
      }
    };

    initMediaPipe();

    // Runtime state matching vanish_cam.py:
    let logAnchor: [number, number, number] | null = null;
    let lastPhase: AppPhase = 'ready';
    let lastGestureCheckMs = 0;
    let lastMaskAtMs = -Infinity;
    let vanishStartedMs = 0;
    let vanishedEnteredMs = 0;
    let readyEnteredMs = 0;
    let lastTriggerAtMs = -Infinity;
    let gestureArmed = true;
    let releasedGestureChecks = 0;
    let activePoseMode: VanishMode | null = null;
    let poseStartMs: number | null = null;
    let lastForeheadPt: { x: number; y: number } | null = null;
    let lastForeheadAtMs = -Infinity;
    let prevHandAnchor: { x: number; y: number } | null = null;
    const GESTURE_COOLDOWN_MS = 2500;
    const PHASE_SETTLE_COOLDOWN_MS = 1800;
    const HOLD_DURATION_BY_MODE: Record<VanishMode, number> = {
      naruto: 320,
      goku: 300,
      normal: 200,
    };

    // Exact port of `_smoke_layer` from `Nicovic17/vanish_webcam/naruto_effect.py`
    const drawNarutoSmokeLayer = (
      targetCtx: CanvasRenderingContext2D,
      anchor: [number, number, number],
      age: number,
      foreground: boolean
    ) => {
      if (age >= NARUTO_EFFECT_SECONDS) return;
      const [cx, cy, size] = anchor;
      const rng = createSeededRng(foreground ? 9127 : 1934);
      const reveal = Math.min(1.0, age / 0.16);
      const fade = Math.max(0.0, 1.0 - Math.max(0.0, age - 0.33) / 1.12);
      const layerAlpha = reveal * fade;
      if (layerAlpha <= 0.01) return;

      const radius = size * (0.27 + 0.54 * Math.min(age, 1.0));
      const count = foreground ? 22 : 18;

      targetCtx.save();
      for (let i = 0; i < count; i++) {
        const angle = rng() * Math.PI * 2;
        const distance = radius * (0.25 + rng() * 0.75);
        const x = cx + Math.cos(angle) * distance;
        const y = cy + Math.sin(angle) * distance * 0.63 - age * size * 0.13;
        const puff = Math.max(4, size * (0.09 + rng() * 0.11) * (0.7 + age * 0.7));
        const rot = ((rng() * 50 - 25) * Math.PI) / 180;

        targetCtx.save();
        targetCtx.translate(x, y);
        targetCtx.rotate(rot);
        targetCtx.beginPath();
        targetCtx.ellipse(0, 0, puff, puff * 0.68, 0, 0, Math.PI * 2);
        if (foreground) {
          targetCtx.fillStyle = `rgba(235, 235, 229, ${(190 / 255) * layerAlpha})`;
        } else {
          targetCtx.fillStyle = `rgba(166, 162, 152, ${(125 / 255) * layerAlpha})`;
        }
        targetCtx.fill();
        targetCtx.restore();
      }

      // Long white winding smoke ribbons from naruto_effect.py
      for (const direction of [-1, 1]) {
        const rx = cx + direction * size * 0.13;
        const ry = cy - size * 0.13 - age * size * 0.09;
        const axisX = Math.max(4, size * (0.3 + age * 0.27));
        const axisY = Math.max(4, size * (0.09 + age * 0.12));

        targetCtx.save();
        targetCtx.translate(rx, ry);
        targetCtx.rotate((direction * 15 * Math.PI) / 180);
        targetCtx.beginPath();
        targetCtx.ellipse(
          0,
          0,
          axisX,
          axisY,
          0,
          (20 * Math.PI) / 180,
          (230 * Math.PI) / 180
        );
        targetCtx.strokeStyle = `rgba(245, 247, 244, ${(220 / 255) * layerAlpha})`;
        targetCtx.lineWidth = Math.max(2, size * 0.025);
        targetCtx.lineCap = 'round';
        targetCtx.stroke();
        targetCtx.restore();
      }

      targetCtx.restore();
    };

    let animId: number;
    const startPerf = performance.now();

    const renderLoop = (now: number) => {
      const {
        mode: curMode,
        phase: curPhase,
        progress: curProgress,
        useRealWebcam: curWebcam,
        mirrorVideo: curMirror,
      } = propsRef.current;

      const timeSec = (now - startPerf) / 1000;

      // Handle switch between Demo and Real Webcam
      if (curWebcam !== lastWebcamState) {
        lastWebcamState = curWebcam;
        backgroundSamples = [];
        logAnchor = null;
        lastMaskAtMs = -Infinity;
        if (curWebcam) {
          bgBuffer.fill(0);
          bgInitialized = false;
          calibrationStartedMs = now;
          lastReportedRemaining = -1;
        } else if (roomLoaded) {
          bgBuffer.set(demoRoomBuffer);
          bgInitialized = true;
          if (personLoaded) {
            maskBuffer.set(demoPhotoMask);
          }
        }
      }

      // Detect phase transitions
      if (curPhase !== lastPhase) {
        if (curPhase === 'calibrating') {
          calibrationStartedMs = now;
          backgroundSamples = [];
          logAnchor = null;
          lastMaskAtMs = -Infinity;
          lastReportedRemaining = -1;
          gestureArmed = false;
          releasedGestureChecks = 0;
          activePoseMode = null;
          poseStartMs = null;
        } else if (curPhase === 'vanishing') {
          vanishStartedMs = now;
          lastTriggerAtMs = now;
          logAnchor = null;
          lastMaskAtMs = -Infinity;
          gestureArmed = false;
          releasedGestureChecks = 0;
          activePoseMode = null;
          poseStartMs = null;
        } else if (curPhase === 'reappearing') {
          lastTriggerAtMs = now;
          gestureArmed = false;
          releasedGestureChecks = 0;
          activePoseMode = null;
          poseStartMs = null;
        } else if (curPhase === 'vanished') {
          vanishedEnteredMs = now;
          gestureArmed = false;
          releasedGestureChecks = 0;
          activePoseMode = null;
          poseStartMs = null;
        } else if (curPhase === 'ready') {
          readyEnteredMs = now;
          logAnchor = null;
          lastMaskAtMs = -Infinity;
          gestureArmed = false;
          releasedGestureChecks = 0;
          activePoseMode = null;
          poseStartMs = null;
        }
        lastPhase = curPhase;
      }

      // 1. Read camera_frame (flipped horizontally if mirrorVideo is true, matching `if not args.no_mirror: camera_frame = cv2.flip(camera_frame, 1)`)
      liveCtx.save();
      if (curWebcam && videoRef.current && videoRef.current.readyState >= 2) {
        if (curMirror) {
          liveCtx.translate(W, 0);
          liveCtx.scale(-1, 1);
        }
        liveCtx.drawImage(videoRef.current, 0, 0, W, H);
      } else if (!curWebcam && roomLoaded) {
        liveCtx.drawImage(roomImg, 0, 0, W, H);
        if (personLoaded && curPhase !== 'calibrating') {
          const breathY = Math.sin(timeSec * 2.1) * 1.8;
          liveCtx.drawImage(personPhotoCanvas, 0, Math.round(breathY));
        }
      }
      liveCtx.restore();

      const liveImageData = liveCtx.getImageData(0, 0, W, H);
      const liveData = liveImageData.data;

      // =========================================================================
      // EXACT PORT OF `run()` LOOP IN `Nicovic17/vanish_webcam/vanish_cam.py`:
      // =========================================================================
      if (curPhase === 'calibrating') {
        const elapsed = (now - calibrationStartedMs) / 1000.0;
        const remaining = Math.max(0, Math.ceil(CALIBRATION_SECONDS - elapsed));
        if (remaining !== lastReportedRemaining) {
          lastReportedRemaining = remaining;
          propsRef.current.onCalibrationTick(remaining);
        }

        // Exact match to vanish_cam.py line 445:
        // `if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS: background_samples.append(camera_frame.copy())`
        if (curWebcam && videoRef.current && videoRef.current.readyState >= 2) {
          if (elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS) {
            if (backgroundSamples.length < 18) {
              backgroundSamples.push(new Uint8ClampedArray(liveData));
            }
          }
        }

        // Exact match to vanish_cam.py line 447:
        // `if elapsed >= CALIBRATION_SECONDS: stack = np.stack(background_samples[-18:], axis=0); background = np.median(stack, axis=0)`
        if (elapsed >= CALIBRATION_SECONDS) {
          if (curWebcam) {
            if (backgroundSamples.length > 0) {
              computeMedianBackground(backgroundSamples, bgBuffer);
            } else {
              bgBuffer.set(liveData);
            }
            backgroundSamples = [];
            bgInitialized = true;
          } else if (roomLoaded) {
            bgBuffer.set(demoRoomBuffer);
            bgInitialized = true;
          }
          propsRef.current.onCalibrationComplete();
        }

        // `display = camera_frame`
        outData.set(liveData);
        ctx.putImageData(outImageData, 0, 0);

        // Countdown overlay during the 4s background acquisition
        ctx.save();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.52)';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = '#F59E0B';
        ctx.font = '700 58px "Outfit", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${remaining}s`, W * 0.5, H * 0.43);
        ctx.fillStyle = '#F8FAFC';
        ctx.font = '600 20px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(
          'Sortez du champ pendant 4 secondes : acquisition du fond vide...',
          W * 0.5,
          H * 0.54
        );
        ctx.fillStyle = '#CBD5E1';
        ctx.font = '400 14px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(
          elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS
            ? 'Capture médiane du fond en cours (0.55s finales)...'
            : 'Le fond sera photographié dans les 0.55 dernières secondes et verrouillé.',
          W * 0.5,
          H * 0.62
        );
        ctx.restore();
      } else if (curPhase === 'vanished' || curPhase === 'ready') {
        if (curPhase === 'vanished') {
          // Exact match to vanish_cam.py lines 455-460:
          // `vanished_frame = background.copy()`
          // `if mode == "naruto" and log_anchor is not None: overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height))`
          outData.set(bgBuffer);
          ctx.putImageData(outImageData, 0, 0);

          if (curMode === 'naruto' && logLoaded) {
            const anchor = logAnchor || substitutionAnchor(maskBuffer, W, H);
            const [lx, ly, lSize] = landedLogAnchor(anchor, H);
            const targetH = Math.max(1, Math.round(lSize));
            const targetW = Math.max(
              1,
              Math.round(
                (targetH * logTrimmedCanvas.width) / Math.max(1, logTrimmedCanvas.height)
              )
            );
            ctx.drawImage(
              logTrimmedCanvas,
              Math.round(lx - targetW / 2),
              Math.round(ly - targetH / 2),
              targetW,
              targetH
            );
          }
        } else {
          // `display = camera_frame` (background is NEVER touched while in `ready`!)
          outData.set(liveData);
          ctx.putImageData(outImageData, 0, 0);
        }

        // Gesture recognition in BOTH `ready` (to vanish) AND `vanished` (to reappear when repeating the gesture!)
        // Enforces a strict cooldown after any trigger AND requires releasing the gesture between triggers.
        const phaseEnteredMs = curPhase === 'vanished' ? vanishedEnteredMs : readyEnteredMs;
        const cooldownReady =
          now - phaseEnteredMs >= PHASE_SETTLE_COOLDOWN_MS &&
          now - lastTriggerAtMs >= GESTURE_COOLDOWN_MS;
        const gestureIntervalMs = 70;

        if (
          curWebcam &&
          now - lastGestureCheckMs >= gestureIntervalMs &&
          handLandmarker &&
          videoRef.current &&
          videoRef.current.readyState >= 2
        ) {
          lastGestureCheckMs = now;
          try {
            const handRes = handLandmarker.detectForVideo(liveWorkCanvas, Math.floor(now));
            const hands2D = handRes.landmarks || [];
            const hands3D = handRes.worldLandmarks || [];

            let foreheadPt: { x: number; y: number } | null = null;
            if (faceLandmarker) {
              try {
                const faceRes = faceLandmarker.detectForVideo(liveWorkCanvas, Math.floor(now));
                const facePts = faceRes.faceLandmarks?.[0];
                if (facePts?.[10]) {
                  const pt10 = facePts[10];
                  const pt151 = facePts[151] || pt10;
                  foreheadPt = {
                    x: (pt10.x + pt151.x) * 0.5,
                    y: (pt10.y + pt151.y) * 0.5,
                  };
                  lastForeheadPt = foreheadPt;
                  lastForeheadAtMs = now;
                } else if (lastForeheadPt && now - lastForeheadAtMs <= 900) {
                  // Keep recent forehead coordinate if hand placed on forehead briefly occludes face
                  foreheadPt = lastForeheadPt;
                }
              } catch {
                if (lastForeheadPt && now - lastForeheadAtMs <= 900) {
                  foreheadPt = lastForeheadPt;
                }
              }
            }

            // Compute primary hand anchor to detect rapid vertical transit (e.g. lifting hand to forehead)
            let curHandAnchor: { x: number; y: number } | null = null;
            if (hands2D.length > 0 && hands2D[0]?.length >= 21) {
              const h0 = hands2D[0];
              curHandAnchor = {
                x: (h0[8].x + h0[12].x) * 0.5,
                y: (h0[8].y + h0[12].y) * 0.5,
              };
            }
            const handMotionDelta =
              curHandAnchor && prevHandAnchor
                ? Math.hypot(
                    curHandAnchor.x - prevHandAnchor.x,
                    curHandAnchor.y - prevHandAnchor.y
                  )
                : 0;
            prevHandAnchor = curHandAnchor;

            // Evaluate all 3 gestures simultaneously (Goku forehead, V-sign Fade Out, Naruto hand seal)
            const match = detectCoexistingGesture(hands3D, hands2D, foreheadPt, W, H);

            if (!match) {
              activePoseMode = null;
              poseStartMs = null;
              propsRef.current.onGokuHoldProgress(0);
              releasedGestureChecks++;
              if (releasedGestureChecks >= 4) {
                gestureArmed = true;
              }
            } else {
              releasedGestureChecks = 0;
              if (cooldownReady && gestureArmed) {
                // If only 1 hand is visible and moving rapidly (e.g. lifting hand toward forehead),
                // reset the Naruto hold timer so it only triggers when held steady in front of the chest
                const isMovingSingleHandNaruto =
                  match.mode === 'naruto' && hands2D.length === 1 && handMotionDelta > 0.045;

                if (
                  activePoseMode !== match.mode ||
                  poseStartMs === null ||
                  isMovingSingleHandNaruto
                ) {
                  activePoseMode = match.mode;
                  poseStartMs = now;
                }
                const requiredHoldMs = HOLD_DURATION_BY_MODE[match.mode] ?? 300;
                const heldRatio = Math.min(1.0, (now - poseStartMs) / requiredHoldMs);
                propsRef.current.onGokuHoldProgress(heldRatio);

                if (heldRatio >= 1.0) {
                  activePoseMode = null;
                  poseStartMs = null;
                  gestureArmed = false;
                  lastTriggerAtMs = now;
                  propsRef.current.onGokuHoldProgress(0);

                  if (match.mode === 'goku') {
                    propsRef.current.onGestureDetected(
                      'goku',
                      curPhase === 'vanished'
                        ? '2 doigts sur le front → Réapparition Shunkan Idō'
                        : '2 doigts sur le front → Téléportation Shunkan Idō'
                    );
                  } else if (match.mode === 'naruto') {
                    propsRef.current.onGestureDetected(
                      'naruto',
                      curPhase === 'vanished'
                        ? 'Signe Naruto refait → Réapparition'
                        : 'Signe Naruto maintenu → Permutation'
                    );
                  } else if (match.mode === 'normal') {
                    propsRef.current.onGestureDetected(
                      'normal',
                      curPhase === 'vanished'
                        ? 'Signe V refait → Réapparition (Fade In)'
                        : 'Signe V détecté → Fade Out'
                    );
                  }
                }
              } else {
                activePoseMode = null;
                poseStartMs = null;
                propsRef.current.onGokuHoldProgress(0);
              }
            }
          } catch {
            // Ignore transient timestamp errors
          }
        }
      } else if (curPhase === 'vanishing' || curPhase === 'reappearing') {
        // Track the silhouette at 30 Hz during fast dissolve; Naruto needs it only while the person is still visible.
        const ageSec = Math.max(0.04, (now - vanishStartedMs) / 1000.0);
        const needsMask =
          curMode === 'naruto'
            ? logAnchor === null || ageSec < 0.13
            : now - lastMaskAtMs >= 33;

        if (needsMask) {
          if (curWebcam && segmenter && videoRef.current && videoRef.current.readyState >= 2) {
            try {
              // Segment `liveWorkCanvas` (which is already mirrored if mirrorVideo=true, matching vanish_cam.py)
              const segResult = segmenter.segmentForVideo(liveWorkCanvas, Math.floor(now));
              if (segResult.confidenceMasks && segResult.confidenceMasks.length > 0) {
                const rawMask = segResult.confidenceMasks[0].getAsFloat32Array();
                const mW = segResult.confidenceMasks[0].width;
                const mH = segResult.confidenceMasks[0].height;
                upsampleAndSmoothMask(rawMask, mW, mH, W, H, false, maskBuffer, maskScratch);
                segResult.confidenceMasks[0].close();
                lastMaskAtMs = now;
              }
            } catch {
              // Keep lastMask
            }
          } else if (!curWebcam && personLoaded) {
            maskBuffer.set(demoPhotoMask);
            lastMaskAtMs = now;
          }
        }

        if (logAnchor === null) {
          logAnchor = substitutionAnchor(maskBuffer, W, H);
        }

        const effectiveProgress =
          curPhase === 'reappearing' ? 1.0 - curProgress : curProgress;

        if (curMode === 'naruto') {
          const anchor = logAnchor || [Math.floor(W / 2), Math.floor(H / 2), Math.floor(H * 0.57)];

          if (curPhase === 'reappearing') {
            // Reappearing on repeating the Naruto hand sign:
            // Smoke puffs forward, the log vanishes in the smoke, and the user reappears!
            const smokeAge = curProgress * 1.15;
            if (curProgress < 0.14) {
              outData.set(bgBuffer);
            } else {
              const appearRatio = Math.min(1.0, (curProgress - 0.14) / 0.18);
              for (let i = 0; i < W * H; i++) {
                const p = i * 4;
                outData[p] = liveData[p] * appearRatio + bgBuffer[p] * (1.0 - appearRatio);
                outData[p + 1] = liveData[p + 1] * appearRatio + bgBuffer[p + 1] * (1.0 - appearRatio);
                outData[p + 2] = liveData[p + 2] * appearRatio + bgBuffer[p + 2] * (1.0 - appearRatio);
                outData[p + 3] = 255;
              }
            }
            ctx.putImageData(outImageData, 0, 0);
            drawNarutoSmokeLayer(ctx, anchor, smokeAge, false);
            drawNarutoSmokeLayer(ctx, anchor, smokeAge, true);
          } else {
            // Exact port of `render_substitution` from `naruto_effect.py` (lines 134-153)
            const age = ageSec;
            if (age >= 0.13) {
              outData.set(bgBuffer);
            } else {
              const survival = Math.max(0.0, 1.0 - age / 0.13);
              alphaCompositeConstantSurvival(liveData, bgBuffer, outData, maskBuffer, survival);
            }
            ctx.putImageData(outImageData, 0, 0);

            drawNarutoSmokeLayer(ctx, anchor, age, false);

            if (logLoaded) {
              const logOpacity = Math.min(1.0, Math.max(0.0, (age - 0.05) / 0.15));
              const scale = 1.15 - Math.min(age, 0.25) * 0.6;
              const [lx, ly, lSize] = fallingLogAnchor(anchor, H, age);
              const targetH = Math.max(1, Math.round(lSize * scale));
              const targetW = Math.max(
                1,
                Math.round(
                  (targetH * logTrimmedCanvas.width) / Math.max(1, logTrimmedCanvas.height)
                )
              );
              ctx.save();
              ctx.globalAlpha = logOpacity;
              ctx.drawImage(
                logTrimmedCanvas,
                Math.round(lx - targetW / 2),
                Math.round(ly - targetH / 2),
                targetW,
                targetH
              );
              ctx.restore();
            }

            drawNarutoSmokeLayer(ctx, anchor, age, true);
          }
        } else if (curMode === 'normal') {
          // Uniform smooth Fade Out / Fade In of the segmented person over the calibrated background
          const eased = easeInOutCubic(effectiveProgress);
          const survival = Math.max(0.0, Math.min(1.0, 1.0 - eased));
          alphaCompositeConstantSurvival(liveData, bgBuffer, outData, maskBuffer, survival);
          ctx.putImageData(outImageData, 0, 0);
        } else if (curMode === 'goku') {
          renderGokuShunkanIdo(
            liveData,
            bgBuffer,
            outData,
            maskBuffer,
            W,
            H,
            effectiveProgress,
            now
          );
          ctx.putImageData(outImageData, 0, 0);
        }
      }

      // Update diagnostic mini-monitors
      if (maskCanvasRef.current) {
        const mCtx = maskCanvasRef.current.getContext('2d');
        if (mCtx) {
          const mImg = mCtx.createImageData(W, H);
          const mData = mImg.data;
          for (let i = 0; i < W * H; i++) {
            const val = Math.round(maskBuffer[i] * 255);
            const p = i * 4;
            mData[p] = Math.round(val * 0.35);
            mData[p + 1] = Math.round(val * 0.92);
            mData[p + 2] = val;
            mData[p + 3] = 255;
          }
          mCtx.putImageData(mImg, 0, 0);
        }
      }

      if (bgCanvasRef.current && bgInitialized) {
        const bCtx = bgCanvasRef.current.getContext('2d');
        if (bCtx) {
          const bImg = new ImageData(new Uint8ClampedArray(bgBuffer), W, H);
          bCtx.putImageData(bImg, 0, 0);
        }
      }

      animId = requestAnimationFrame(renderLoop);
    };

    animId = requestAnimationFrame(renderLoop);

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      segmenter?.close();
      handLandmarker?.close();
      faceLandmarker?.close();
    };
  }, [canvasRef, maskCanvasRef, bgCanvasRef, videoRef, onMediaPipeStatusChange]);

  return (
    <div className="relative w-full aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-xl">
      <video ref={videoRef} className="hidden" playsInline muted autoPlay />
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        data-testid="vanish-output-canvas"
        data-phase={phase}
        data-mode={mode}
        className="w-full h-full object-cover block"
      />
    </div>
  );
};
