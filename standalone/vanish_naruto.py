#!/usr/bin/env python3
"""
VanishCam — Effet "Substitution Naruto" (Kawarimi no Jutsu)
Script 100 % autonome (macOS & Windows)

Fonctionnalités :
- Résolution caméra native adaptative (utilise la vraie qualité de votre webcam sans forcer)
- Lecture caméra Zero-Lag en thread dédié
- Détection du signe de mains Naruto (Mudra Tigre/Bélier devant le torse) via MediaPipe
- Double couche de fumée anime + chute physique d'une bûche de substitution avec rebond
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
            print("Puis lancez :\n  python vanish_naruto.py")
        else:
            print("  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt\n")
            print("Puis lancez :\n  python3 vanish_naruto.py")
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
mp_face_module = None
mp_selfie_module = None

try:
    import mediapipe.python.solutions as mp_solutions
    mp_hands_module = mp_solutions.hands
    mp_face_module = mp_solutions.face_mesh
    mp_selfie_module = mp_solutions.selfie_segmentation
except Exception:
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            mp_hands_module = mp.solutions.hands
            mp_face_module = mp.solutions.face_mesh
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


class _TasksFaceResult:
    def __init__(self, multi_face_landmarks):
        self.multi_face_landmarks = multi_face_landmarks


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


class _TasksFaceModule:
    class FaceMesh:
        def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                     min_detection_confidence=0.35, min_tracking_confidence=0.35):
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            self._mp = mp
            model_path = _ensure_model_asset(
                "face_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)

        def process(self, rgb_frame):
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            res = self._detector.detect(mp_img)
            f_lms = [_TasksLandmarkList(lm) for lm in res.face_landmarks] if res.face_landmarks else None
            return _TasksFaceResult(f_lms)


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
        mp_face_module = _TasksFaceModule
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
# DÉTECTION DU SIGNE DE MAINS NARUTO (MUDRA DEVANT LE TORSE)
# ============================================================================

def _dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def has_naruto_mudra_finger_posture(h, scale, strict_single_merged_hand=False):
    index_len = _dist2d(h[0], h[8])
    middle_len = _dist2d(h[0], h[12])
    index_pip_len = _dist2d(h[0], h[6])
    middle_pip_len = _dist2d(h[0], h[10])

    index_extended = (
        index_len > index_pip_len * 1.08
        and index_len > scale * 1.15
        and h[8][1] < h[6][1]
    )
    middle_extended = (
        middle_len > middle_pip_len * 1.08
        and middle_len > scale * 1.15
        and (h[12][1] < h[10][1] or _dist2d(h[8], h[12]) < scale * 0.45)
    )
    if not index_extended or not middle_extended:
        return False

    req_reach = 0.75 if strict_single_merged_hand else 0.65
    vertical_reach = (
        (h[0][1] - h[8][1]) > scale * req_reach
        and (h[0][1] - h[12][1]) > scale * req_reach
    )
    if not vertical_reach:
        return False

    if strict_single_merged_hand:
        index_vertical = abs(h[8][0] - h[5][0]) <= max(1.0, h[5][1] - h[8][1]) * 0.75
        middle_vertical = abs(h[12][0] - h[9][0]) <= max(1.0, h[9][1] - h[12][1]) * 0.75
        if not index_vertical or not middle_vertical:
            return False

    max_tip_gap = scale * (0.58 if strict_single_merged_hand else 0.65)
    tip_gap = _dist2d(h[8], h[12])
    tips_together = tip_gap < max_tip_gap
    not_spread_v = tip_gap <= max(_dist2d(h[5], h[9]) * 1.35, scale * 0.52)
    if not tips_together or not not_spread_v:
        return False

    ring_len = _dist2d(h[0], h[16])
    pinky_len = _dist2d(h[0], h[20])
    blade_top_y = min(h[8][1], h[12][1])

    ring_folded = (
        h[16][1] > blade_top_y + scale * 0.20
        or ring_len < index_len * 0.90
        or ring_len < _dist2d(h[0], h[14]) * 1.15
    )
    pinky_folded = (
        h[20][1] > blade_top_y + scale * 0.20
        or pinky_len < middle_len * 0.90
        or pinky_len < _dist2d(h[0], h[18]) * 1.15
    )
    return ring_folded and pinky_folded


def joined_hands_from_landmarks(hands_norm, width, height):
    if not hands_norm or len(hands_norm) < 2:
        return False
    valid_hands = [h for h in hands_norm[:2] if h and len(h) == 21]
    if len(valid_hands) != 2:
        return False

    pts = [[(p[0] * width, p[1] * height) for p in hand] for hand in valid_hands]
    palm_indices = [0, 5, 9, 13, 17]
    tip_indices = [4, 8, 12, 16, 20]

    palms = [
        (
            sum(hand[idx][0] for idx in palm_indices) / 5.0,
            sum(hand[idx][1] for idx in palm_indices) / 5.0,
        )
        for hand in pts
    ]
    fingertips = [
        (
            sum(hand[idx][0] for idx in tip_indices) / 5.0,
            sum(hand[idx][1] for idx in tip_indices) / 5.0,
        )
        for hand in pts
    ]

    for palm in palms:
        if (
            palm[0] < width * 0.14
            or palm[0] > width * 0.86
            or palm[1] < height * 0.16
            or palm[1] > height * 0.88
        ):
            return False

    scales = [max(_dist2d(hand[5], hand[17]), _dist2d(hand[0], hand[9])) for hand in pts]
    min_scale = min(scales[0], scales[1])
    max_scale = max(scales[0], scales[1])
    if min_scale < 12.0 or max_scale > min_scale * 2.1:
        return False

    size = (scales[0] + scales[1]) * 0.5
    up0 = (pts[0][0][1] - pts[0][9][1]) > scales[0] * 0.35
    up1 = (pts[1][0][1] - pts[1][9][1]) > scales[1] * 0.35
    if not up0 or not up1:
        return False

    palms_joined = _dist2d(palms[0], palms[1]) < size * 1.65
    tips_joined = (
        _dist2d(fingertips[0], fingertips[1]) < size * 1.85
        or _dist2d(pts[0][8], pts[1][8]) < size * 1.15
        or _dist2d(pts[0][12], pts[1][12]) < size * 1.15
        or _dist2d(pts[0][5], pts[1][5]) < size * 1.25
    )
    if not palms_joined or not tips_joined:
        return False

    mudra0 = has_naruto_mudra_finger_posture(pts[0], scales[0], False)
    mudra1 = has_naruto_mudra_finger_posture(pts[1], scales[1], False)

    not_open_palm0 = (
        pts[0][16][1] > min(pts[0][8][1], pts[0][12][1]) + scales[0] * 0.22
        and pts[0][20][1] > min(pts[0][8][1], pts[0][12][1]) + scales[0] * 0.22
    )
    not_open_palm1 = (
        pts[1][16][1] > min(pts[1][8][1], pts[1][12][1]) + scales[1] * 0.22
        and pts[1][20][1] > min(pts[1][8][1], pts[1][12][1]) + scales[1] * 0.22
    )
    return (mudra0 or mudra1) and not_open_palm0 and not_open_palm1


def is_naruto_hand_seal(hands_norm, width, height, face_forehead_point=None):
    if not hands_norm:
        return False
    if len(hands_norm) >= 2 and joined_hands_from_landmarks(hands_norm, width, height):
        return True

    valid_hands = [
        [(p[0] * width, p[1] * height) for p in hand]
        for hand in hands_norm
        if hand and len(hand) == 21
    ]
    if not valid_hands:
        return False

    palm_indices = [0, 5, 9, 13, 17]
    if len(valid_hands) >= 2:
        h0, h1 = valid_hands[0], valid_hands[1]
        palm0 = (sum(h0[idx][0] for idx in palm_indices) / 5.0, sum(h0[idx][1] for idx in palm_indices) / 5.0)
        palm1 = (sum(h1[idx][0] for idx in palm_indices) / 5.0, sum(h1[idx][1] for idx in palm_indices) / 5.0)
        scale0 = max(_dist2d(h0[5], h0[17]), _dist2d(h0[0], h0[9]))
        scale1 = max(_dist2d(h1[5], h1[17]), _dist2d(h1[0], h1[9]))
        size = (scale0 + scale1) * 0.5

        if has_naruto_mudra_finger_posture(h0, scale0, False) or has_naruto_mudra_finger_posture(h1, scale1, False):
            hands_close = (
                _dist2d(palm0, palm1) < size * 1.75
                or _dist2d(h0[8], h1[8]) < size * 1.15
                or _dist2d(h0[7], h1[7]) < size * 1.10
                or _dist2d(h0[12], h1[12]) < size * 1.15
            )
            if hands_close:
                return True

    for h in valid_hands:
        scale = max(_dist2d(h[5], h[17]), _dist2d(h[0], h[9]))
        if scale < 14.0:
            continue
        palm = (sum(h[idx][0] for idx in palm_indices) / 5.0, sum(h[idx][1] for idx in palm_indices) / 5.0)
        mid_tip = ((h[8][0] + h[12][0]) * 0.5, (h[8][1] + h[12][1]) * 0.5)

        if palm[0] < width * 0.12 or palm[0] > width * 0.88 or mid_tip[0] < width * 0.12 or mid_tip[0] > width * 0.88:
            continue

        min_chest_tip_y = (face_forehead_point[1] + 0.04) * height if face_forehead_point is not None else height * 0.18
        min_chest_palm_y = (face_forehead_point[1] + 0.12) * height if face_forehead_point is not None else height * 0.28

        if mid_tip[1] < min_chest_tip_y or palm[1] < min_chest_palm_y or palm[1] > height * 0.95:
            continue
        if has_naruto_mudra_finger_posture(h, scale, True):
            return True

    return False


# ============================================================================
# MOTEUR DE SUBSTITUTION NARUTO (KAWARIMI NO JUTSU)
# ============================================================================

class VanishNarutoEngine:
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
        self.vanish_duration = 1.45
        self.reappear_duration = 0.50

        self.saved_naruto_anchor = None
        self.cached_mask = None
        self.cached_bbox = None
        self.last_mask_time = 0.0

        # Anti-rebond et maintien du geste (0.30s)
        self.last_trigger_time = 0.0
        self.trigger_cooldown = 1.65
        self.phase_settle_cooldown = 0.55
        self.pose_start_time = None
        self.required_hold = 0.30
        self.gesture_armed = True
        self.released_checks = 0
        self.prev_hand_anchor = None
        self.last_forehead_pt = None
        self.last_forehead_time = 0.0

        # Chargement du sprite de la bûche
        self.log_img = None
        self._load_log_sprite()

        # MediaPipe
        self.mp_hands = mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.45
        )
        self.mp_face = mp_face_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.mp_selfie = mp_selfie_module.SelfieSegmentation(model_selection=1)

        # Thread asynchrone
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

    def _load_log_sprite(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join(script_dir, "assets", "substitution_log.png"), "assets/substitution_log.png"):
            if os.path.exists(p):
                loaded = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if loaded is not None:
                    if len(loaded.shape) == 3 and loaded.shape[2] == 4:
                        alpha = loaded[:, :, 3]
                        ay, ax = np.where(alpha > 20)
                        if len(ax) > 0 and len(ay) > 0:
                            self.log_img = loaded[int(np.min(ay)):int(np.max(ay)) + 1, int(np.min(ax)):int(np.max(ax)) + 1]
                        else:
                            self.log_img = loaded
                    else:
                        self.log_img = loaded
                    print(f"🪵 Sprite bûche chargé ({self.log_img.shape[1]}x{self.log_img.shape[0]} px)")
                    break

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
            time.sleep(max(0.005, 0.070 - elapsed))

    def _check_gesture_sync(self, frame_bgr):
        now = time.time()
        ai_w, ai_h = 960, 540
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (ai_w, ai_h)), cv2.COLOR_BGR2RGB)

        hands_res = self.mp_hands.process(small_rgb)
        hands_2d = []
        if hands_res.multi_hand_landmarks:
            for h_lm in hands_res.multi_hand_landmarks:
                hands_2d.append([(p.x, p.y) for p in h_lm.landmark])

        forehead_pt = None
        face_res = self.mp_face.process(small_rgb)
        if face_res.multi_face_landmarks:
            flm = face_res.multi_face_landmarks[0].landmark
            forehead_pt = ((flm[10].x + flm[151].x) * 0.5, (flm[10].y + flm[151].y) * 0.5)
            self.last_forehead_pt = forehead_pt
            self.last_forehead_time = now
        elif self.last_forehead_pt is not None and (now - self.last_forehead_time) <= 0.90:
            forehead_pt = self.last_forehead_pt

        cur_anchor = None
        if hands_2d and len(hands_2d[0]) >= 21:
            cur_anchor = ((hands_2d[0][8][0] + hands_2d[0][12][0]) * 0.5, (hands_2d[0][8][1] + hands_2d[0][12][1]) * 0.5)
        motion_delta = _dist2d(cur_anchor, self.prev_hand_anchor) if (cur_anchor and self.prev_hand_anchor) else 0.0
        self.prev_hand_anchor = cur_anchor

        detected = is_naruto_hand_seal(hands_2d, ai_w, ai_h, forehead_pt)
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
            is_moving_single = (len(hands_2d) == 1 and motion_delta > 0.045)
            if self.pose_start_time is None or is_moving_single:
                self.pose_start_time = now
            if (now - self.pose_start_time) >= self.required_hold:
                self.pose_start_time = None
                self.gesture_armed = False
                self.last_trigger_time = now
                return True
        else:
            self.pose_start_time = None
        return False

    def play_sound(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in (os.path.join(script_dir, "assets", filename), os.path.join("assets", filename)):
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
        self.saved_naruto_anchor = None
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
            print("✅ Calibrage terminé ! Joignez vos mains en Mudra Naruto 🙏 pour vous substituer.")

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

    def compute_substitution_anchor(self, mask, w, h):
        sub = mask[::4, ::4]
        ay, ax = np.where(sub > 0.40)
        if len(ax) < 8:
            return (w // 2, h // 2, int(h * 0.57))
        min_x, max_x = int(np.min(ax)) * 4, int(np.max(ax)) * 4
        min_y, max_y = int(np.min(ay)) * 4, int(np.max(ay)) * 4
        sil_w, sil_h = max_x - min_x, max_y - min_y
        size = int(max(h * 0.24, min(h * 0.70, sil_h * 0.58)))
        margin = size // 3
        cx = int(max(margin, min(w - margin, min_x + sil_w * 0.5)))
        cy = int(max(margin, min(h - margin, min_y + sil_h * 0.52)))
        return (cx, cy, size)

    def compute_falling_log_anchor(self, anchor, h, age):
        cx, cy, size = anchor
        floor_y = max(cy, h - int(round(size * 0.48)))
        distance = floor_y - cy
        if distance <= 0 or age <= 0.18:
            return (cx, cy, size)
        elapsed = age - 0.18
        gravity = h * 3.2
        fall = gravity * (elapsed ** 2) * 0.5
        if fall < distance:
            return (cx, cy + int(round(fall)), size)
        landed_at = math.sqrt((2.0 * distance) / gravity)
        bounce_age = elapsed - landed_at
        bounce = size * 0.075 * math.exp(-5.0 * bounce_age) * abs(math.sin(11.0 * bounce_age))
        return (cx, floor_y - int(round(bounce)), size)

    def _render_naruto_smoke_layer(self, canvas, anchor, age, foreground=False):
        if age >= 1.45:
            return
        cx, cy, size = anchor
        rng = np.random.RandomState(9127 if foreground else 1934)
        reveal = min(1.0, age / 0.16)
        fade = max(0.0, 1.0 - max(0.0, age - 0.33) / 1.12)
        layer_alpha = reveal * fade
        if layer_alpha <= 0.01:
            return

        h, w = canvas.shape[:2]
        radius = size * (0.27 + 0.54 * min(age, 1.0))
        max_extent = int(radius + size * 0.45)

        rx1 = max(0, int(cx - max_extent))
        ry1 = max(0, int(cy - max_extent - age * size * 0.15))
        rx2 = min(w, int(cx + max_extent))
        ry2 = min(h, int(cy + max_extent))
        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = canvas[ry1:ry2, rx1:rx2]
        overlay = roi.copy()
        count = 22 if foreground else 18
        color_bgr = (229, 235, 235) if foreground else (152, 162, 166)
        base_alpha = (190.0 / 255.0) * layer_alpha if foreground else (125.0 / 255.0) * layer_alpha

        for _ in range(count):
            angle = rng.rand() * math.pi * 2
            distance = radius * (0.25 + rng.rand() * 0.75)
            px = int(cx + math.cos(angle) * distance) - rx1
            py = int(cy + math.sin(angle) * distance * 0.63 - age * size * 0.13) - ry1
            puff = max(4, int(size * (0.09 + rng.rand() * 0.11) * (0.7 + age * 0.7)))
            rot_deg = int(rng.rand() * 50 - 25)
            cv2.ellipse(overlay, (px, py), (puff, max(2, int(puff * 0.68))), rot_deg, 0, 360, color_bgr, -1)

        ribbon_color = (244, 247, 245)
        for direction in [-1, 1]:
            rx = int(cx + direction * size * 0.13) - rx1
            ry = int(cy - size * 0.13 - age * size * 0.09) - ry1
            axis_x = max(4, int(size * (0.3 + age * 0.27)))
            axis_y = max(4, int(size * (0.09 + age * 0.12)))
            thick = max(2, int(size * 0.025))
            cv2.ellipse(overlay, (rx, ry), (axis_x, axis_y), direction * 15, 20, 230, ribbon_color, thick)

        cv2.addWeighted(overlay, base_alpha, roi, 1.0 - base_alpha, 0, roi)

    def _render_naruto_log(self, output, anchor, h, age):
        if self.log_img is None:
            lx, ly, l_size = self.compute_falling_log_anchor(anchor, h, age)
            bw, bh = int(l_size * 0.65), int(l_size * 0.95)
            cv2.rectangle(output, (lx - bw // 2, ly - bh // 2), (lx + bw // 2, ly + bh // 2), (42, 68, 128), -1)
            cv2.rectangle(output, (lx - bw // 2, ly - bh // 2), (lx + bw // 2, ly + bh // 2), (20, 35, 75), 3)
            return

        log_h_native, log_w_native = self.log_img.shape[:2]
        aspect_ratio = log_w_native / float(log_h_native)
        log_opacity = min(1.0, max(0.0, (age - 0.05) / 0.15))
        if log_opacity <= 0.02:
            return

        scale = 1.15 - min(age, 0.25) * 0.6
        lx, ly, l_size = self.compute_falling_log_anchor(anchor, h, age)
        target_h = max(1, int(round(l_size * scale)))
        target_w = max(1, int(round(target_h * aspect_ratio)))

        w_out = output.shape[1]
        x1, y1 = lx - target_w // 2, ly - target_h // 2
        x2, y2 = x1 + target_w, y1 + target_h

        src_x1, src_y1 = max(0, -x1), max(0, -y1)
        src_x2, src_y2 = target_w - max(0, x2 - w_out), target_h - max(0, y2 - h)
        dst_x1, dst_y1 = max(0, x1), max(0, y1)
        dst_x2, dst_y2 = min(w_out, x2), min(h, y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1 or src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        resized = cv2.resize(self.log_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        crop_log = resized[src_y1:src_y2, src_x1:src_x2]

        if crop_log.shape[2] == 4:
            alpha = (crop_log[:, :, 3].astype(np.float32) * np.float32(log_opacity / 255.0))[:, :, np.newaxis]
            bgr = crop_log[:, :, :3]
            roi = output[dst_y1:dst_y2, dst_x1:dst_x2]
            output[dst_y1:dst_y2, dst_x1:dst_x2] = (bgr * alpha + roi * (np.float32(1.0) - alpha)).astype(np.uint8)
        else:
            output[dst_y1:dst_y2, dst_x1:dst_x2] = crop_log

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
                self.saved_naruto_anchor = self.compute_substitution_anchor(self.cached_mask, w, h)
            print("🪵 Substitution Naruto (Kawarimi no Jutsu) déclenchée !")
            self.play_sound("vanish.mp3")
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
            self.play_sound("whoosh.wav")

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
            self.saved_naruto_anchor = None
            return frame_bgr

        if self.phase == 'vanished':
            output = bg.copy()
            anchor = self.saved_naruto_anchor or (w // 2, h // 2, int(h * 0.57))
            self._render_naruto_log(output, anchor, h, 1.45)
            return output

        if self.cached_mask is None or (elapsed < 0.13 and (now - self.last_mask_time) >= 0.045):
            self.cached_mask = self._get_fast_smooth_mask(frame_bgr, w, h)
            self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
            self.last_mask_time = now

        if self.saved_naruto_anchor is None:
            self.saved_naruto_anchor = self.compute_substitution_anchor(self.cached_mask, w, h)

        anchor = self.saved_naruto_anchor

        if self.phase == 'vanishing':
            if elapsed < 0.13:
                survival = max(0.0, 1.0 - elapsed / 0.13)
                output = self._blend_person_roi(frame_bgr, bg, self.cached_mask, self.cached_bbox, survival)
            else:
                output = bg.copy()

            self._render_naruto_smoke_layer(output, anchor, elapsed, foreground=False)
            self._render_naruto_log(output, anchor, h, elapsed)
            self._render_naruto_smoke_layer(output, anchor, elapsed, foreground=True)

            if elapsed >= self.vanish_duration:
                self.phase = 'vanished'
                self.phase_entered_time = now
            return output

        if self.phase == 'reappearing':
            progress = min(1.0, elapsed / self.reappear_duration)
            appear_ratio = min(1.0, max(0.0, (progress - 0.14) / 0.35))
            output = self._blend_person_roi(frame_bgr, bg, self.cached_mask, self.cached_bbox, appear_ratio)

            smoke_age = progress * 1.15
            self._render_naruto_smoke_layer(output, anchor, smoke_age, foreground=False)
            self._render_naruto_smoke_layer(output, anchor, smoke_age, foreground=True)

            if progress >= 1.0:
                self.phase = 'ready'
                self.phase_entered_time = now
                self.saved_naruto_anchor = None
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
    parser = argparse.ArgumentParser(description="VanishCam - Effet Substitution Naruto")
    parser.add_argument("--cam", type=int, default=0, help="Index de la webcam (défaut: 0)")
    parser.add_argument("--width", type=int, default=None, help="Largeur optionnelle")
    parser.add_argument("--height", type=int, default=None, help="Hauteur optionnelle")
    parser.add_argument("--fps", type=int, default=30, help="FPS cible")
    args = parser.parse_args()

    print("=" * 60)
    print("🪵 VANISHCAM — Effet Indépendant : SUBSTITUTION NARUTO")
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
    engine = VanishNarutoEngine(actual_w, actual_h)
    fast_cam = ZeroLagCamera(cap)

    vcam = None
    if HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"✅ Caméra virtuelle active : {vcam.device}")
        except Exception:
            print("ℹ️  Capturez la fenêtre 'VanishCam - Naruto' dans OBS Studio")

    print("\nCommandes :")
    print("  🙏 Geste     : Mains jointes en Mudra Naruto devant le torse")
    print("  [Espace]   : Recalibrer le décor vide (4 secondes)")
    print("  [2] ou [N] : Déclencher manuellement la Substitution Naruto")
    print("  [C]        : Changer de webcam")
    print("  [R]        : Réapparaître immédiatement")
    print("  [Q]        : Quitter")
    print("-" * 60)

    window_name = "VanishCam - Naruto"
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
            elif key in (ord('2'), ord('n'), ord('N')):
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
