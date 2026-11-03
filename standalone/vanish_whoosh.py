#!/usr/bin/env python3
"""
VanishCam — Effet "Whoosh" (Disparition en Fondu par Signe V)
Script 100 % autonome (macOS & Windows)

Fonctionnalités :
- Résolution caméra native adaptative (utilise la vraie qualité de votre webcam sans forcer)
- Lecture caméra Zero-Lag en thread dédié
- Détection du Signe V de la victoire (Index + Majeur écartés) via MediaPipe Hands
- Disparition et réapparition en fondu fluide vers le décor vide calibré
- Sortie directe vers OBS Virtual Camera (pyvirtualcam) ou Capture de Fenêtre OBS
"""

import sys
import os
import time
import math
import threading
import argparse
import site
import glob


def _ensure_venv_and_deps():
    """Charge automatiquement les paquets du dossier .venv local (si présent) sans relancer de sous-processus."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, ".venv")
    if os.path.isdir(venv_dir):
        if os.name == "nt":
            site_dirs = [os.path.join(venv_dir, "Lib", "site-packages")]
        else:
            site_dirs = glob.glob(os.path.join(venv_dir, "lib", "python*", "site-packages"))
        for d in site_dirs:
            if os.path.isdir(d) and d not in sys.path:
                site.addsitedir(d)

    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError:
        print("\n========================================================")
        print("  [!] Dépendances Python non installées")
        print("========================================================")
        print("Copiez-collez cette commande dans votre Terminal (1 seule fois) :\n")
        if os.name == "nt":
            print("  python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt\n")
            print("Puis lancez :\n  python vanish_whoosh.py")
        else:
            print("  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt\n")
            print("Puis lancez :\n  python3 vanish_whoosh.py")
        print("========================================================\n")
        sys.exit(1)


_ensure_venv_and_deps()


import numpy as np
import cv2


class ZeroLagCamera:
    """Lit la webcam en continu en arrière-plan pour garantir 0 ms de latence."""
    def __init__(self, cap):
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()


# Chargement de MediaPipe (compatible mp.solutions ET MediaPipe 0.10.30+ / 1.x Tasks API)
mp_hands_module = None
mp_selfie_module = None

try:
    import mediapipe.python.solutions as mp_solutions
    mp_hands_module = mp_solutions.hands
    mp_selfie_module = mp_solutions.selfie_segmentation
except Exception:
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            mp_hands_module = mp.solutions.hands
            mp_selfie_module = mp.solutions.selfie_segmentation
    except Exception:
        pass


class _TasksLandmarkList:
    def __init__(self, landmark):
        self.landmark = landmark


class _TasksHandsResult:
    def __init__(self, multi_hand_landmarks, multi_hand_world_landmarks=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_hand_world_landmarks = multi_hand_world_landmarks


class _TasksSelfieResult:
    def __init__(self, segmentation_mask):
        self.segmentation_mask = segmentation_mask


def _ensure_model_asset(filename, url):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "assets", filename),
        os.path.join(script_dir, filename),
        os.path.join("assets", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1024:
            return c
    target = os.path.join(script_dir, "assets", filename)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"📥 Téléchargement du modèle MediaPipe {filename}...")
    import urllib.request
    urllib.request.urlretrieve(url, target)
    return target


class _TasksHandsModule:
    class Hands:
        def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1,
                     min_detection_confidence=0.35, min_tracking_confidence=0.45):
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            self._mp = mp
            model_path = _ensure_model_asset(
                "hand_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)

        def process(self, rgb_frame):
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            res = self._detector.detect(mp_img)
            h_lms = [_TasksLandmarkList(lm) for lm in res.hand_landmarks] if res.hand_landmarks else None
            hw_lms = [_TasksLandmarkList(lm) for lm in res.hand_world_landmarks] if getattr(res, "hand_world_landmarks", None) else None
            return _TasksHandsResult(h_lms, hw_lms)


class _TasksSelfieModule:
    class SelfieSegmentation:
        def __init__(self, model_selection=1):
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            self._mp = mp
            model_path = _ensure_model_asset(
                "selfie_segmenter_landscape.tflite",
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
            )
            options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)

        def process(self, rgb_frame):
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            res = self._segmenter.segment(mp_img)
            if res.confidence_masks:
                idx = 1 if len(res.confidence_masks) > 1 else 0
                mask = np.squeeze(res.confidence_masks[idx].numpy_view()).astype(np.float32)
            else:
                h, w = rgb_frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)
            return _TasksSelfieResult(mask)


if mp_hands_module is None:
    try:
        import mediapipe as _mp_check  # noqa: F401
        mp_hands_module = _TasksHandsModule
        mp_selfie_module = _TasksSelfieModule
    except Exception as e:
        print(f"❌ Erreur lors du chargement de MediaPipe : {e}")
        print("💡 Solution : pip install -r requirements.txt")
        sys.exit(1)

try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

try:
    import pygame
    pygame.mixer.init()
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False


# ============================================================================
# DÉTECTION DU SIGNE "V" DE LA VICTOIRE (3D & 2D)
# ============================================================================

def _dist3d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def victory_landmark_score(points):
    """
    Reconnaît l'index et le majeur tendus et écartés en "V" tandis que
    l'annulaire et l'auriculaire sont repliés.
    """
    if not points or len(points) != 21:
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []

    bases = [5, 9, 13, 17]
    for index, base in enumerate(bases):
        f0 = points[base]
        f1 = points[base + 1]
        f2 = points[base + 2]
        f3 = points[base + 3]

        length = _dist3d(f0, f1) + _dist3d(f1, f2) + _dist3d(f2, f3)
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue

        ratio = _dist3d(f3, f0) / length
        tip_distance = _dist3d(f3, points[0])
        joint_distance = _dist3d(f1, points[0])

        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.1)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    reference_length = min(lengths[0], lengths[1])
    for index, base in ((2, 13), (3, 17)):
        reach = _dist3d(points[base + 3], points[base])
        folded[index] = folded[index] or (reach < reference_length * 0.65)

    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0

    separation = _dist3d(points[8], points[12])
    mcp_separation = _dist3d(points[5], points[9])
    dip_separation = _dist3d(points[7], points[11])

    if (
        separation < reference_length * 0.32
        or separation <= mcp_separation * 1.18
        or separation <= dip_separation * 1.02
    ):
        return 0.0

    return max(0.0, min(1.0, min(straightness[0], straightness[1])))


def is_victory_sign(world_hands, image_hands, width=960, height=540):
    if world_hands:
        for landmarks in world_hands:
            if victory_landmark_score(landmarks) >= 0.62:
                return True
    if image_hands:
        for landmarks in image_hands:
            pts3d = [(p[0] * width, p[1] * height, 0.0) for p in landmarks]
            if victory_landmark_score(pts3d) >= 0.62:
                return True
    return False


# ============================================================================
# MOTEUR DE DISPARITION EN FONDU (WHOOSH)
# ============================================================================

class VanishWhooshEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Calibration (4 secondes)
        self.calibrating = True
        self.calibration_start = time.time()
        self.calibration_duration = 4.0
        self.calib_frames = []
        self.median_background = None

        # États : 'ready', 'vanishing', 'vanished', 'reappearing'
        self.phase = 'ready'
        self.phase_start_time = 0.0
        self.phase_entered_time = time.time()
        self.fade_out_duration = 1.15
        self.fade_in_duration = 0.50

        self.cached_mask = None
        self.cached_bbox = None
        self.last_mask_time = 0.0

        # Anti-rebond et maintien du geste (0.22s)
        self.last_trigger_time = 0.0
        self.trigger_cooldown = 1.65
        self.phase_settle_cooldown = 0.55
        self.pose_start_time = None
        self.required_hold = 0.22
        self.gesture_armed = True
        self.released_checks = 0

        # MediaPipe Hands + Selfie Segmentation (pas besoin de FaceMesh ici)
        self.mp_hands = mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.45
        )
        self.mp_selfie = mp_selfie_module.SelfieSegmentation(model_selection=1)

        # Thread asynchrone de détection du geste
        self._lock = threading.Lock()
        self._gesture_frame = None
        self._pending_trigger = False
        self._running = True
        self._thread = threading.Thread(target=self._gesture_worker, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=0.4)
        except Exception:
            pass

    def _gesture_worker(self):
        while self._running:
            frame_to_check = None
            with self._lock:
                if self._gesture_frame is not None:
                    frame_to_check = self._gesture_frame
                    self._gesture_frame = None

            if frame_to_check is None:
                time.sleep(0.010)
                continue

            t0 = time.time()
            try:
                if self._check_gesture_sync(frame_to_check):
                    with self._lock:
                        self._pending_trigger = True
            except Exception:
                pass

            elapsed = time.time() - t0
            time.sleep(max(0.005, 0.065 - elapsed))

    def _check_gesture_sync(self, frame_bgr):
        now = time.time()
        ai_w, ai_h = 960, 540
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (ai_w, ai_h)), cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(small_rgb)

        hands_2d = []
        hands_3d = []
        if res.multi_hand_landmarks:
            for h_lm in res.multi_hand_landmarks:
                hands_2d.append([(p.x, p.y) for p in h_lm.landmark])
        if getattr(res, 'multi_hand_world_landmarks', None):
            for hw_lm in res.multi_hand_world_landmarks:
                hands_3d.append([(p.x, p.y, p.z) for p in hw_lm.landmark])

        detected = is_victory_sign(hands_3d, hands_2d, ai_w, ai_h)
        if not detected:
            self.pose_start_time = None
            self.released_checks += 1
            if self.released_checks >= 4:
                self.gesture_armed = True
            return False

        self.released_checks = 0
        cooldown_ready = (
            (now - self.phase_entered_time) >= self.phase_settle_cooldown
            and (now - self.last_trigger_time) >= self.trigger_cooldown
        )
        if cooldown_ready and self.gesture_armed:
            if self.pose_start_time is None:
                self.pose_start_time = now
            if (now - self.pose_start_time) >= self.required_hold:
                self.pose_start_time = None
                self.gesture_armed = False
                self.last_trigger_time = now
                return True
        else:
            self.pose_start_time = None
        return False

    def play_whoosh_sound(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in (os.path.join(script_dir, "assets", "whoosh.wav"), "assets/whoosh.wav"):
            if os.path.exists(candidate):
                if sys.platform == 'darwin':
                    try:
                        import subprocess
                        subprocess.Popen(['afplay', candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                    except Exception:
                        pass
                if HAS_AUDIO:
                    try:
                        pygame.mixer.music.load(candidate)
                        pygame.mixer.music.play()
                    except Exception:
                        pass
                return

    def start_calibration(self):
        self.calibrating = True
        self.calibration_start = time.time()
        self.calib_frames = []
        self.median_background = None
        self.phase = 'ready'
        self.phase_entered_time = time.time()
        print("📸 Calibrage du décor démarré... Sortez du champ pendant 4 secondes.")

    def update_calibration(self, frame_bgr):
        now = time.time()
        elapsed = now - self.calibration_start
        if elapsed >= (self.calibration_duration - 0.75):
            if not hasattr(self, '_last_sample') or (now - self._last_sample) >= 0.08:
                self._last_sample = now
                self.calib_frames.append(frame_bgr.copy())
                if len(self.calib_frames) > 9:
                    self.calib_frames.pop(0)

        if elapsed >= self.calibration_duration:
            if self.calib_frames:
                stack = np.stack(self.calib_frames, axis=0)
                self.median_background = np.median(stack, axis=0).astype(np.uint8)
            else:
                self.median_background = frame_bgr.copy()
            self.calibrating = False
            self.phase = 'ready'
            self.phase_entered_time = now
            print("✅ Calibrage terminé ! Faites un Signe V ✌️ devant la caméra pour disparaître.")

    def _get_fast_smooth_mask(self, frame_bgr, w, h):
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        seg_res = self.mp_selfie.process(small_rgb)
        raw_mask = seg_res.segmentation_mask.astype(np.float32)
        blurred_low = cv2.GaussianBlur(raw_mask, (7, 7), 1.4)
        stretched_low = np.clip((blurred_low - 0.12) / 0.68, 0.0, 1.0)
        return cv2.resize(stretched_low, (w, h), interpolation=cv2.INTER_LINEAR)

    def _compute_bbox_from_mask(self, mask, w, h):
        sub = mask[::4, ::4]
        ay, ax = np.where(sub > 0.05)
        if len(ax) > 8:
            min_x = max(0, int(np.min(ax)) * 4 - 8)
            max_x = min(w, int(np.max(ax)) * 4 + 8)
            min_y = max(0, int(np.min(ay)) * 4 - 8)
            max_y = min(h, int(np.max(ay)) * 4 + 8)
            if max_x > min_x + 20 and max_y > min_y + 20:
                return (min_x, max_x, min_y, max_y)
        return (int(w * 0.15), int(w * 0.85), int(h * 0.05), int(h * 0.98))

    def _blend_person_roi(self, frame_bgr, bg_bgr, smooth_mask, bbox, survival):
        output = bg_bgr.copy()
        if survival <= 0.005:
            return output
        min_x, max_x, min_y, max_y = bbox
        roi_f = frame_bgr[min_y:max_y, min_x:max_x]
        roi_b = output[min_y:max_y, min_x:max_x]
        roi_m = (smooth_mask[min_y:max_y, min_x:max_x] * np.float32(survival))[:, :, np.newaxis]
        output[min_y:max_y, min_x:max_x] = (roi_f * roi_m + roi_b * (np.float32(1.0) - roi_m)).astype(np.uint8)
        return output

    def trigger_toggle(self, current_frame_bgr=None):
        now = time.time()
        self.last_trigger_time = now
        self.pose_start_time = None
        self.gesture_armed = False

        if self.phase == 'ready':
            self.phase = 'vanishing'
            self.phase_start_time = now
            self.phase_entered_time = now
            if current_frame_bgr is not None:
                h, w = current_frame_bgr.shape[:2]
                self.cached_mask = self._get_fast_smooth_mask(current_frame_bgr, w, h)
                self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
                self.last_mask_time = now
            print("✌️ Disparition Whoosh déclenchée !")
            self.play_whoosh_sound()
        elif self.phase == 'vanished':
            self.phase = 'reappearing'
            self.phase_start_time = now
            self.phase_entered_time = now
            if current_frame_bgr is not None:
                h, w = current_frame_bgr.shape[:2]
                self.cached_mask = self._get_fast_smooth_mask(current_frame_bgr, w, h)
                self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
                self.last_mask_time = now
            print("👤 Réapparition déclenchée !")
            self.play_whoosh_sound()

    def process_frame(self, frame_bgr):
        now = time.time()
        h, w = frame_bgr.shape[:2]

        if self.calibrating:
            self.update_calibration(frame_bgr)
            return frame_bgr

        if self.phase in ('ready', 'vanished'):
            should_trigger = False
            with self._lock:
                if self._pending_trigger:
                    should_trigger = True
                    self._pending_trigger = False
                elif self._gesture_frame is None:
                    self._gesture_frame = frame_bgr
            if should_trigger:
                self.trigger_toggle(frame_bgr)

        bg = self.median_background if self.median_background is not None else frame_bgr
        elapsed = now - self.phase_start_time

        if self.phase == 'ready':
            return frame_bgr
        if self.phase == 'vanished':
            return bg.copy()

        if self.cached_mask is None or (now - self.last_mask_time) >= 0.045:
            self.cached_mask = self._get_fast_smooth_mask(frame_bgr, w, h)
            self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
            self.last_mask_time = now

        if self.phase == 'vanishing':
            progress = min(1.0, elapsed / self.fade_out_duration)
            output = self._blend_person_roi(frame_bgr, bg, self.cached_mask, self.cached_bbox, max(0.0, 1.0 - progress))
            if progress >= 1.0:
                self.phase = 'vanished'
                self.phase_entered_time = now
            return output

        if self.phase == 'reappearing':
            progress = min(1.0, elapsed / self.fade_in_duration)
            output = self._blend_person_roi(frame_bgr, bg, self.cached_mask, self.cached_bbox, progress)
            if progress >= 1.0:
                self.phase = 'ready'
                self.phase_entered_time = now
            return output

        return frame_bgr


def negotiate_native_camera_resolution(cap, requested_w=None, requested_h=None):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if requested_w is not None and requested_h is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_h)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, sample_frame = cap.read()
    if not ret or sample_frame is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, sample_frame = cap.read()

    if ret and sample_frame is not None:
        actual_h, actual_w = sample_frame.shape[:2]
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    return actual_w, actual_h


def open_webcam_by_index(idx):
    if sys.platform == 'darwin':
        c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if c.isOpened():
            return c
    c = cv2.VideoCapture(idx)
    return c if c.isOpened() else None


def main():
    parser = argparse.ArgumentParser(description="VanishCam - Effet Whoosh (Signe V)")
    parser.add_argument("--cam", type=int, default=0, help="Index de la webcam (défaut: 0)")
    parser.add_argument("--width", type=int, default=None, help="Largeur optionnelle")
    parser.add_argument("--height", type=int, default=None, help="Hauteur optionnelle")
    parser.add_argument("--fps", type=int, default=30, help="FPS cible")
    args = parser.parse_args()

    print("=" * 60)
    print("✌️  VANISHCAM — Effet Indépendant : WHOOSH (Signe V)")
    print("=" * 60)

    cap = None
    active_cam_idx = args.cam
    for idx in [args.cam] + [i for i in (0, 1, 2) if i != args.cam]:
        c = open_webcam_by_index(idx)
        if c is not None:
            cap = c
            active_cam_idx = idx
            print(f"📹 Webcam connectée (index {idx})")
            break

    if cap is None or not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        sys.exit(1)

    actual_w, actual_h = negotiate_native_camera_resolution(cap, args.width, args.height)
    print(f"✨ Résolution native : {actual_w}x{actual_h}")
    engine = VanishWhooshEngine(actual_w, actual_h)
    fast_cam = ZeroLagCamera(cap)

    vcam = None
    if HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"✅ Caméra virtuelle active : {vcam.device}")
        except Exception:
            print("ℹ️  Capturez la fenêtre 'VanishCam - Whoosh' dans OBS Studio")

    print("\nCommandes :")
    print("  ✌️  Geste     : Signe V de la victoire pour disparaître / réapparaître")
    print("  [Espace]   : Recalibrer le décor vide (4 secondes)")
    print("  [1] ou [V] : Déclencher manuellement le fondu")
    print("  [C]        : Changer de webcam")
    print("  [R]        : Réapparaître immédiatement")
    print("  [Q]        : Quitter")
    print("-" * 60)

    window_name = "VanishCam - Whoosh"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = fast_cam.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            frame = cv2.flip(frame, 1)
            out_frame = engine.process_frame(frame)

            if vcam is not None:
                vcam.send(out_frame)
            cv2.imshow(window_name, out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                engine.start_calibration()
            elif key in (ord('1'), ord('v'), ord('V')):
                engine.trigger_toggle(frame)
            elif key in (ord('c'), ord('C')):
                for offset in (1, 2, 3):
                    next_idx = (active_cam_idx + offset) % 4
                    new_cap = open_webcam_by_index(next_idx)
                    if new_cap is not None:
                        fast_cam.release()
                        active_cam_idx = next_idx
                        actual_w, actual_h = negotiate_native_camera_resolution(new_cap, args.width, args.height)
                        fast_cam = ZeroLagCamera(new_cap)
                        print(f"🔄 Caméra index {active_cam_idx} ({actual_w}x{actual_h})")
                        engine.start_calibration()
                        break
            elif key in (ord('r'), ord('R')):
                if engine.phase == 'vanished':
                    engine.trigger_toggle(frame)
    finally:
        engine.close()
        fast_cam.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
