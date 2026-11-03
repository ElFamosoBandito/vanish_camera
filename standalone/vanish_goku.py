#!/usr/bin/env python3
"""
VanishCam — Effet "Téléportation Goku" (Dragon Ball Z Shunkan Idō)
Script 100 % autonome (macOS & Windows)

Fonctionnalités :
- Résolution caméra native adaptative (utilise la vraie qualité de votre webcam sans forcer)
- Lecture caméra Zero-Lag en thread dédié
- Détection du geste de Goku (Index + Majeur serrés sur le front) via MediaPipe Hands + FaceMesh
- Rendu vectoriel DBZ : hachures d'encre verticales et découpe de la silhouette en bandes
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
            print("Puis lancez :\n  python vanish_goku.py")
        else:
            print("  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt\n")
            print("Puis lancez :\n  python3 vanish_goku.py")
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
# DÉTECTION DU GESTE DE GOKU (2 DOIGTS SUR LE FRONT)
# ============================================================================

def is_goku_forehead_gesture(hands_norm, face_forehead_point):
    """
    Détecte la pose du Déplacement Instantané (Shunkan Idō) :
    - Index (8) et Majeur (12) tendus et serrés ensemble
    - Annulaire (16) et Auriculaire (20) repliés
    - Extrémité des doigts posée sur le front
    """
    if not hands_norm:
        return False

    for hand in hands_norm:
        if not hand or len(hand) < 21:
            continue
        index_tip = hand[8]
        middle_tip = hand[12]
        mid_point = (
            (index_tip[0] + middle_tip[0]) * 0.5,
            (index_tip[1] + middle_tip[1]) * 0.5,
        )

        scale = max(
            0.04,
            math.hypot(hand[5][0] - hand[17][0], hand[5][1] - hand[17][1]),
            math.hypot(hand[0][0] - hand[9][0], hand[0][1] - hand[9][1]),
        )

        fingers_standing_up = (
            index_tip[1] < hand[6][1]
            and middle_tip[1] < hand[10][1]
            and hand[0][1] > mid_point[1] + scale * 0.55
        )
        if not fingers_standing_up:
            continue

        tip_gap = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
        mcp_gap = math.hypot(hand[5][0] - hand[9][0], hand[5][1] - hand[9][1])
        tips_together = tip_gap < max(0.095, scale * 0.55) and tip_gap <= mcp_gap * 1.25
        if not tips_together:
            continue

        index_reach = math.hypot(index_tip[0] - hand[0][0], index_tip[1] - hand[0][1])
        middle_reach = math.hypot(middle_tip[0] - hand[0][0], middle_tip[1] - hand[0][1])
        ring_reach = math.hypot(hand[16][0] - hand[0][0], hand[16][1] - hand[0][1])
        pinky_reach = math.hypot(hand[20][0] - hand[0][0], hand[20][1] - hand[0][1])

        ring_folded = (
            hand[16][1] > mid_point[1] + scale * 0.18
            or ring_reach < index_reach * 0.88
        )
        pinky_folded = (
            hand[20][1] > mid_point[1] + scale * 0.18
            or pinky_reach < middle_reach * 0.88
        )
        if not ring_folded or not pinky_folded:
            continue

        if face_forehead_point is not None:
            dist_to_forehead = math.hypot(
                mid_point[0] - face_forehead_point[0],
                mid_point[1] - face_forehead_point[1],
            )
            on_forehead_height = (
                mid_point[1] <= face_forehead_point[1] + 0.115
                and mid_point[1] >= face_forehead_point[1] - 0.14
            )
            if dist_to_forehead < 0.155 and on_forehead_height:
                return True
        else:
            in_upper_head_zone = (
                0.25 <= mid_point[0] <= 0.75
                and 0.05 <= mid_point[1] <= 0.32
            )
            if in_upper_head_zone:
                return True

    return False


# ============================================================================
# MOTEUR DE TÉLÉPORTATION GOKU (SHUNKAN IDO)
# ============================================================================

class VanishGokuEngine:
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
        self.vanish_duration = 1.15
        self.reappear_duration = 0.50

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
        self.last_forehead_pt = None
        self.last_forehead_time = 0.0

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

        detected = is_goku_forehead_gesture(hands_2d, forehead_pt)
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

    def play_goku_sound(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in (os.path.join(script_dir, "assets", "goku.wav"), "assets/goku.wav"):
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
            print("✅ Calibrage terminé ! Posez 2 doigts sur votre front ⚡ pour vous téléporter.")

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

    def render_goku_shunkan_ido(self, frame_bgr, bg_bgr, mask, bbox, p, now_ms):
        """
        Rendu DBZ Shunkan Idō Ultra-Optimisé (60+ FPS constant sans lag) :
        - Découpe en bandes verticales avec révélation du fond calibré
        - Smear vertical matériel accéléré par OpenCV (cv2.blur / cv2.addWeighted)
        - Bordures d'encre cel-shaded noires nettes (style manga DBZ)
        - Hachures d'encre dynamiques haute vitesse
        - Zéro allocation de grille 2D lourde, calculs 100% vectorisés en ROI
        """
        h, w = frame_bgr.shape[:2]
        if p >= 0.94:
            return bg_bgr.copy()

        min_x, max_x, min_y, max_y = bbox
        pad_x, pad_y = 16, 32
        rx1, rx2 = max(0, min_x - pad_x), min(w, max_x + pad_x)
        ry1, ry2 = max(0, min_y - pad_y), min(h, max_y + pad_y)

        rw, rh = rx2 - rx1, ry2 - ry1
        if rw <= 10 or rh <= 10:
            return bg_bgr.copy()

        output = bg_bgr.copy()
        roi_f = frame_bgr[ry1:ry2, rx1:rx2]
        roi_b = bg_bgr[ry1:ry2, rx1:rx2]
        roi_m = mask[ry1:ry2, rx1:rx2]

        sil_h = max(40, max_y - min_y)
        anime_step = int(now_ms / 48)
        scale_x = max(1.0, w / 960.0)

        # 1. Smear vertical anime ultra-rapide via OpenCV hardware SIMD
        # (Remplace les 3 indexations 2D lentes qui faisaient chuter les FPS)
        smear_strength = float(
            min(0.62, (p / 0.15) * 0.55) if p < 0.42
            else max(0.18, 0.55 - (p - 0.42) * 0.8)
        )
        blur_k = max(3, int(round((7 + p * 16) * scale_x))) | 1
        smeared_f = cv2.blur(roi_f, (1, blur_k))
        smeared = cv2.addWeighted(roi_f, 1.0 - smear_strength, smeared_f, smear_strength, 0)

        # 2. Découpage en bandes verticales DBZ & décalages par bande
        base_strips = [15, 9, 19, 7, 13, 17, 8, 22, 10, 14, 6, 18, 11, 16]
        strip_widths = [max(4, int(round(sw * scale_x))) for sw in base_strips]
        cur_x = 0
        strip_idx = 0

        slits = []
        borders = []

        while cur_x < rw:
            base_w = strip_widths[strip_idx % len(strip_widths)]
            strip_w = min(rw - cur_x, base_w)
            sx1 = cur_x
            sx2 = cur_x + strip_w

            h1 = abs(math.sin(strip_idx * 12.9898 + 4.1414) * 43758.5453) % 1.0
            h2 = abs(math.sin(strip_idx * 78.233 + 1.618) * 19341.123) % 1.0
            h3 = abs(math.sin((strip_idx + anime_step * 0.35) * 31.71) * 27183.81) % 1.0

            open_at = 0.12 + h1 * 0.52
            slide_dir = -1 if strip_idx % 2 == 0 else 1
            strip_slide = int(round(slide_dir * (3 + h2 * 15) * scale_x * (max(0.0, (p - 0.08) / 0.8) ** 1.2)))

            # Décalage vertical léger de la bande (anime offset)
            if strip_slide != 0 and abs(strip_slide) < rh:
                smeared[:, sx1:sx2] = np.roll(smeared[:, sx1:sx2], strip_slide, axis=0)

            open_ratio = 0.0 if p <= open_at else min(1.0, (p - open_at) / max(0.14, 0.84 - open_at))
            global_collapse = (p - 0.72) / 0.16 if p > 0.72 else 0.0
            effective_open_ratio = min(1.0, open_ratio * 0.92 + global_collapse)

            if effective_open_ratio > 0.04:
                slit_w = int(round(strip_w * effective_open_ratio))
                if slit_w > 0:
                    mid = (sx1 + sx2) * 0.5
                    slit_x1 = max(0, int(round(mid - slit_w * 0.5)))
                    slit_x2 = min(rw, int(round(mid + slit_w * 0.5)))

                    if slit_x2 > slit_x1:
                        v_span_ratio = min(1.0, effective_open_ratio * 1.45)
                        if v_span_ratio >= 0.88:
                            sy1, sy2 = 0, rh
                        else:
                            anchor_y = (min_y - ry1) + sil_h * (0.25 + 0.5 * h3)
                            half_span = sil_h * (0.28 + 0.65 * v_span_ratio)
                            sy1 = max(0, int(anchor_y - half_span))
                            sy2 = min(rh, int(anchor_y + half_span))

                        if sy2 > sy1:
                            slits.append((slit_x1, slit_x2, sy1, sy2))
                            if p >= 0.14:
                                bw = max(1, int(round(scale_x)))
                                borders.append((max(0, slit_x1 - bw), slit_x1, sy1, sy2))
                                borders.append((slit_x2, min(rw, slit_x2 + bw), sy1, sy2))

            cur_x += strip_w
            strip_idx += 1

        # 3. Composition ultra-rapide personne / décor (ROI local uniquement)
        m3d = roi_m[:, :, np.newaxis]
        person_blend = (smeared * m3d + roi_b * (np.float32(1.0) - m3d)).astype(np.uint8)

        # 4. Révélation instantanée du décor dans les fentes ouvertes
        for sx1, sx2, sy1, sy2 in slits:
            person_blend[sy1:sy2, sx1:sx2] = roi_b[sy1:sy2, sx1:sx2]

        # 5. Bordures d'encre de manga DBZ le long des découpes verticales
        ink_color = (26, 18, 14)
        for bx1, bx2, by1, by2 in borders:
            if bx2 > bx1 and by2 > by1:
                sub_mask = roi_m[by1:by2, bx1:bx2]
                ink_region = person_blend[by1:by2, bx1:bx2]
                ink_region[sub_mask > 0.30] = ink_color

        # 6. Lignes de vitesse verticales d'encre DBZ (tracées directement en C)
        num_speed_lines = 45 if p < 0.4 else (30 if p < 0.8 else 15)
        rng = np.random.RandomState((4099 + anime_step * 131) & 0xFFFFFFFF)
        xs = rng.randint(1, rw - 1, size=num_speed_lines)
        lens = (sil_h * (0.12 + rng.rand(num_speed_lines) * 0.38)).astype(np.int32)
        y_starts = rng.randint(0, max(1, rh - 30), size=num_speed_lines)

        for l in range(num_speed_lines):
            lx = int(xs[l])
            ly1 = int(y_starts[l])
            ly2 = min(rh - 1, ly1 + int(lens[l]))
            if ly2 <= ly1:
                continue
            mid_y = (ly1 + ly2) // 2
            if roi_m[mid_y, lx] < 0.25:
                continue
            is_white = (l % 3 == 0) if p < 0.42 else (l % 7 == 0)
            color = (238, 248, 250) if is_white else (24, 16, 12)
            thick = 2 if (l % 5 == 0 and scale_x > 1.2) else 1
            cv2.line(person_blend, (lx, ly1), (lx, ly2), color, thick)

        output[ry1:ry2, rx1:rx2] = person_blend
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
            print("⚡ Téléportation Goku (Shunkan Idō) déclenchée !")
            self.play_goku_sound()
        elif self.phase == 'vanished':
            self.phase = 'reappearing'
            self.phase_start_time = now
            self.phase_entered_time = now
            if current_frame_bgr is not None:
                h, w = current_frame_bgr.shape[:2]
                self.cached_mask = self._get_fast_smooth_mask(current_frame_bgr, w, h)
                self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
                self.last_mask_time = now
            print("⚡ Réapparition Goku déclenchée !")
            self.play_goku_sound()

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

        if self.cached_mask is None or self.cached_bbox is None:
            self.cached_mask = self._get_fast_smooth_mask(frame_bgr, w, h)
            self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
            self.last_mask_time = now

        if self.phase == 'vanishing':
            progress = min(1.0, elapsed / self.vanish_duration)
            output = self.render_goku_shunkan_ido(frame_bgr, bg, self.cached_mask, self.cached_bbox, progress, int(now * 1000))
            if progress >= 0.94:
                self.phase = 'vanished'
                self.phase_entered_time = now
            return output

        if self.phase == 'reappearing':
            progress = min(1.0, elapsed / self.reappear_duration)
            output = self.render_goku_shunkan_ido(frame_bgr, bg, self.cached_mask, self.cached_bbox, 1.0 - progress, int(now * 1000))
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
    parser = argparse.ArgumentParser(description="VanishCam - Effet Téléportation Goku")
    parser.add_argument("--cam", type=int, default=0, help="Index de la webcam (défaut: 0)")
    parser.add_argument("--width", type=int, default=None, help="Largeur optionnelle")
    parser.add_argument("--height", type=int, default=None, help="Hauteur optionnelle")
    parser.add_argument("--fps", type=int, default=30, help="FPS cible")
    args = parser.parse_args()

    print("=" * 60)
    print("⚡ VANISHCAM — Effet Indépendant : TÉLÉPORTATION GOKU")
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
    engine = VanishGokuEngine(actual_w, actual_h)
    fast_cam = ZeroLagCamera(cap)

    vcam = None
    if HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"✅ Caméra virtuelle active : {vcam.device}")
        except Exception:
            print("ℹ️  Capturez la fenêtre 'VanishCam - Goku' dans OBS Studio")

    print("\nCommandes :")
    print("  ⚡ Geste     : 2 doigts (Index + Majeur) posés sur le front")
    print("  [Espace]   : Recalibrer le décor vide (4 secondes)")
    print("  [3] ou [G] : Déclencher manuellement la Téléportation Goku")
    print("  [C]        : Changer de webcam")
    print("  [R]        : Réapparaître immédiatement")
    print("  [Q]        : Quitter")
    print("-" * 60)

    window_name = "VanishCam - Goku"
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
            elif key in (ord('3'), ord('g'), ord('G')):
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
