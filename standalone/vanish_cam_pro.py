#!/usr/bin/env python3
"""
VanishCam Pro — Real-Time Anime & VFX Virtual Camera (macOS & Windows)

Features:
- Adaptive native webcam resolution (automatically uses whatever resolution your camera offers)
- Zero-latency background frame reader thread
- Asynchronous MediaPipe Hands (Full Model) + FaceMesh gesture engine
- 3 Simultaneous Hand Gestures (no manual mode switching required):
  1. Smooth Fade Out / Fade In: Victory "V" sign
  2. Naruto Substitution Jutsu (Kawarimi): Clasped Tiger/Ram hand seal in front of chest
  3. Goku Instant Transmission (Shunkan Ido): Index + Middle fingers on forehead
- 4-second temporal median background calibration ([Space] key)
- Direct output to OBS Virtual Camera (via pyvirtualcam) or OBS Window Capture
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
            print("Puis lancez :\n  python vanish_cam_pro.py")
        else:
            print("  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt\n")
            print("Puis lancez :\n  python3 vanish_cam_pro.py")
        print("========================================================\n")
        sys.exit(1)


_ensure_venv_and_deps()


import numpy as np
import cv2


class ZeroLagCamera:
    """
    Continuously drains the OS/OpenCV camera FIFO buffer in a background thread
    so .read() always returns the freshest real-time frame with zero latency.
    """
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


# Import MediaPipe (supports both legacy mp.solutions and MediaPipe 0.10.30+ / 1.x Tasks API)
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
    def __init__(self, multi_hand_landmarks, multi_hand_world_landmarks):
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
# 3D & 2D HAND GESTURE RECOGNITION ENGINE
# ============================================================================

def _dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist3d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def victory_landmark_score(points):
    """
    Recognizes index and middle fingers extended in a V-sign while ring and pinky
    are folded, invariant to 3D hand rotation and scale.
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


def classify_victory_from_hands(world_hands, image_hands, width, height):
    """
    Evaluates V-sign score across both 3D world landmarks and 2D image landmarks.
    """
    if world_hands:
        for landmarks in world_hands:
            score = victory_landmark_score(landmarks)
            if score > 0.0:
                return score

    if image_hands:
        for landmarks in image_hands:
            pts3d = [(p[0] * width, p[1] * height, 0.0) for p in landmarks]
            score = victory_landmark_score(pts3d)
            if score > 0.0:
                return score

    return 0.0


def has_naruto_mudra_finger_posture(h, scale, strict_single_merged_hand=False):
    """
    Checks whether a hand in 2D pixel space forms the vertical Naruto Tiger/Ram seal blade:
    - Index (8) and Middle (12) extended straight upward and pressed together
    - Ring (16) and Pinky (20) folded downward
    """
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
    """
    Recognizes two nearby, vertically aligned hands clasped together in front of the chest.
    """
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

    scales = [
        max(_dist2d(hand[5], hand[17]), _dist2d(hand[0], hand[9]))
        for hand in pts
    ]
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


def is_goku_forehead_gesture(hands_norm, face_forehead_point):
    """
    Detects Goku's Instant Transmission (Shunkan Ido) pose:
    - Index (8) and Middle (12) extended and pressed together
    - Ring (16) and Pinky (20) folded
    - Fingertips touching the forehead (with upper-head fallback if hand occludes face)
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


def is_naruto_hand_seal(hands_norm, width, height, face_forehead_point=None):
    """
    Recognizes the Naruto Substitution Hand Seal (2 clasped hands or merged interlocked mudra).
    """
    if not hands_norm:
        return False
    if is_goku_forehead_gesture(hands_norm, face_forehead_point):
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
        h0 = valid_hands[0]
        h1 = valid_hands[1]
        palm0 = (
            sum(h0[idx][0] for idx in palm_indices) / 5.0,
            sum(h0[idx][1] for idx in palm_indices) / 5.0,
        )
        palm1 = (
            sum(h1[idx][0] for idx in palm_indices) / 5.0,
            sum(h1[idx][1] for idx in palm_indices) / 5.0,
        )
        scale0 = max(_dist2d(h0[5], h0[17]), _dist2d(h0[0], h0[9]))
        scale1 = max(_dist2d(h1[5], h1[17]), _dist2d(h1[0], h1[9]))
        size = (scale0 + scale1) * 0.5

        blade0 = has_naruto_mudra_finger_posture(h0, scale0, False)
        blade1 = has_naruto_mudra_finger_posture(h1, scale1, False)

        if blade0 or blade1:
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

        palm = (
            sum(h[idx][0] for idx in palm_indices) / 5.0,
            sum(h[idx][1] for idx in palm_indices) / 5.0,
        )
        mid_tip = (
            (h[8][0] + h[12][0]) * 0.5,
            (h[8][1] + h[12][1]) * 0.5,
        )

        if (
            palm[0] < width * 0.12
            or palm[0] > width * 0.88
            or mid_tip[0] < width * 0.12
            or mid_tip[0] > width * 0.88
        ):
            continue

        min_chest_tip_y = (
            (face_forehead_point[1] + 0.04) * height
            if face_forehead_point is not None
            else height * 0.18
        )
        min_chest_palm_y = (
            (face_forehead_point[1] + 0.12) * height
            if face_forehead_point is not None
            else height * 0.28
        )

        if mid_tip[1] < min_chest_tip_y or palm[1] < min_chest_palm_y or palm[1] > height * 0.95:
            continue

        if has_naruto_mudra_finger_posture(h, scale, True):
            return True

    return False


def detect_coexisting_gesture(world_hands, image_hands, face_forehead_point, width=960, height=540):
    """
    Evaluates all 3 vanish gestures simultaneously:
    1. Goku ('goku')
    2. Victory V-Sign ('normal')
    3. Naruto Tiger/Ram Seal ('naruto')
    """
    if not image_hands:
        return None

    if is_goku_forehead_gesture(image_hands, face_forehead_point):
        return 'goku'

    v_score = classify_victory_from_hands(world_hands, image_hands, width, height)
    if v_score >= 0.62:
        return 'normal'

    if is_naruto_hand_seal(image_hands, width, height, face_forehead_point):
        return 'naruto'

    return None


# ============================================================================
# REAL-TIME VFX & SEGMENTATION ENGINE
# ============================================================================

class VanishCamEngine:
    def __init__(self, camera_idx=0, width=1920, height=1080, fps=30):
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.fps = fps

        # Calibration state (4.0s total countdown, median sampled over final 0.75s)
        self.calibrating = True
        self.calibration_start = time.time()
        self.calibration_duration = 4.0
        self.calib_frames = []
        self.median_background = None

        # Vanish state: 'ready', 'vanishing', 'vanished', 'reappearing'
        self.phase = 'ready'
        self.active_mode = None  # 'normal', 'naruto', 'goku'
        self.phase_start_time = 0.0
        self.phase_entered_time = time.time()

        self.mode_durations = {
            'normal': 1.15,
            'naruto': 1.45,
            'goku': 1.15,
            'reappearing': 0.50
        }

        self.saved_naruto_anchor = None
        self.cached_mask = None
        self.cached_bbox = None
        self.last_mask_time = 0.0

        # Anti-false trigger & hold timers
        self.last_trigger_time = 0.0
        self.trigger_cooldown = 1.65
        self.phase_settle_cooldown = 0.55
        self.pose_start_time = None
        self.candidate_mode = None
        self.required_hold = {'normal': 0.22, 'naruto': 0.30, 'goku': 0.30}
        self.gesture_armed = True
        self.released_gesture_checks = 0
        self.prev_hand_anchor = None

        # Forehead coordinate persistence (900ms occlusion buffer)
        self.last_forehead_pt = None
        self.last_forehead_time = 0.0

        # Naruto substitution log sprite
        self.log_img = None
        self._load_log_sprite()

        # MediaPipe Solutions (Full Hand model for maximum accuracy)
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

        # Asynchronous Gesture Detection Worker Thread (zero impact on 1080p video FPS)
        self._gesture_lock = threading.Lock()
        self._gesture_frame = None
        self._pending_triggered_mode = None
        self._gesture_running = True
        self._gesture_thread = threading.Thread(target=self._gesture_worker_loop, daemon=True)
        self._gesture_thread.start()

    def close(self):
        self._gesture_running = False
        try:
            self._gesture_thread.join(timeout=0.4)
        except Exception:
            pass

    def _gesture_worker_loop(self):
        while self._gesture_running:
            frame_to_check = None
            with self._gesture_lock:
                if self._gesture_frame is not None:
                    frame_to_check = self._gesture_frame
                    self._gesture_frame = None

            if frame_to_check is None:
                time.sleep(0.010)
                continue

            t0 = time.time()
            try:
                mode = self._evaluate_gestures_sync(frame_to_check)
                if mode is not None:
                    with self._gesture_lock:
                        self._pending_triggered_mode = mode
            except Exception:
                pass

            elapsed = time.time() - t0
            sleep_for = max(0.005, 0.070 - elapsed)
            time.sleep(sleep_for)

    def _evaluate_gestures_sync(self, frame_bgr):
        now = time.time()
        ai_w, ai_h = 960, 540
        small_bgr = cv2.resize(frame_bgr, (ai_w, ai_h), interpolation=cv2.INTER_LINEAR)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

        hands_res = self.mp_hands.process(small_rgb)
        hands_2d = []
        hands_3d = []

        if hands_res.multi_hand_landmarks:
            for h_lm in hands_res.multi_hand_landmarks:
                hands_2d.append([(p.x, p.y) for p in h_lm.landmark])

        if getattr(hands_res, 'multi_hand_world_landmarks', None):
            for hw_lm in hands_res.multi_hand_world_landmarks:
                hands_3d.append([(p.x, p.y, p.z) for p in hw_lm.landmark])

        forehead_pt = None
        face_res = self.mp_face.process(small_rgb)
        if face_res.multi_face_landmarks:
            flm = face_res.multi_face_landmarks[0].landmark
            forehead_pt = (
                (flm[10].x + flm[151].x) * 0.5,
                (flm[10].y + flm[151].y) * 0.5,
            )
            self.last_forehead_pt = forehead_pt
            self.last_forehead_time = now
        elif self.last_forehead_pt is not None and (now - self.last_forehead_time) <= 0.90:
            forehead_pt = self.last_forehead_pt

        cur_hand_anchor = None
        if len(hands_2d) > 0 and len(hands_2d[0]) >= 21:
            h0 = hands_2d[0]
            cur_hand_anchor = (
                (h0[8][0] + h0[12][0]) * 0.5,
                (h0[8][1] + h0[12][1]) * 0.5,
            )

        hand_motion_delta = (
            _dist2d(cur_hand_anchor, self.prev_hand_anchor)
            if (cur_hand_anchor is not None and self.prev_hand_anchor is not None)
            else 0.0
        )
        self.prev_hand_anchor = cur_hand_anchor

        matched_mode = detect_coexisting_gesture(hands_3d, hands_2d, forehead_pt, ai_w, ai_h)

        if matched_mode is None:
            self.candidate_mode = None
            self.pose_start_time = None
            self.released_gesture_checks += 1
            if self.released_gesture_checks >= 4:
                self.gesture_armed = True
            return None

        self.released_gesture_checks = 0
        cooldown_ready = (
            (now - self.phase_entered_time) >= self.phase_settle_cooldown
            and (now - self.last_trigger_time) >= self.trigger_cooldown
        )

        if cooldown_ready and self.gesture_armed:
            is_moving_single_hand_naruto = (
                matched_mode == 'naruto' and len(hands_2d) == 1 and hand_motion_delta > 0.045
            )
            if (
                self.candidate_mode != matched_mode
                or self.pose_start_time is None
                or is_moving_single_hand_naruto
            ):
                self.candidate_mode = matched_mode
                self.pose_start_time = now

            req_hold = self.required_hold.get(matched_mode, 0.30)
            if (now - self.pose_start_time) >= req_hold:
                self.candidate_mode = None
                self.pose_start_time = None
                self.gesture_armed = False
                self.last_trigger_time = now
                return matched_mode
        else:
            self.candidate_mode = None
            self.pose_start_time = None

        return None

    def _load_log_sprite(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "assets", "substitution_log.png"),
            os.path.join(script_dir, "substitution_log.png"),
            "assets/substitution_log.png",
            "substitution_log.png",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                loaded = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if loaded is not None:
                    if len(loaded.shape) == 3 and loaded.shape[2] == 4:
                        alpha = loaded[:, :, 3]
                        active_y, active_x = np.where(alpha > 20)
                        if len(active_x) > 0 and len(active_y) > 0:
                            min_x, max_x = int(np.min(active_x)), int(np.max(active_x))
                            min_y, max_y = int(np.min(active_y)), int(np.max(active_y))
                            self.log_img = loaded[min_y:max_y + 1, min_x:max_x + 1]
                        else:
                            self.log_img = loaded
                    else:
                        self.log_img = loaded
                    print(f"🪵 Sprite bûche chargé ({self.log_img.shape[1]}x{self.log_img.shape[0]} px)")
                    break

    def play_sound(self, sound_type):
        sound_map = {
            'naruto': "assets/vanish.mp3",
            'goku': "assets/goku.wav",
            'normal': "assets/whoosh.wav",
            'return': "assets/whoosh.wav",
        }
        target = sound_map.get(sound_type, "assets/vanish.mp3")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, target),
            target,
        ]
        sound_file = None
        for c in candidates:
            if os.path.exists(c):
                sound_file = c
                break

        if not sound_file:
            return

        if sys.platform == 'darwin':
            try:
                import subprocess
                subprocess.Popen(['afplay', sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass

        if HAS_AUDIO:
            try:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
            except Exception:
                pass

    def start_calibration(self):
        self.calibrating = True
        self.calibration_start = time.time()
        self.calib_frames = []
        self.median_background = None
        self.phase = 'ready'
        self.phase_entered_time = time.time()
        self.active_mode = None
        self.saved_naruto_anchor = None
        self.cached_mask = None
        self.cached_bbox = None
        print("📸 Calibrage du décor démarré... Sortez du champ de la caméra pendant 4 secondes.")

    def update_calibration(self, frame_bgr):
        now = time.time()
        elapsed = now - self.calibration_start
        if elapsed >= (self.calibration_duration - 0.75):
            if not hasattr(self, '_last_calib_sample') or (now - self._last_calib_sample) >= 0.08:
                self._last_calib_sample = now
                self.calib_frames.append(frame_bgr.copy())
                if len(self.calib_frames) > 9:
                    self.calib_frames.pop(0)

        remaining = max(0.0, self.calibration_duration - elapsed)
        if remaining <= 0:
            if len(self.calib_frames) > 0:
                stack = np.stack(self.calib_frames, axis=0)
                self.median_background = np.median(stack, axis=0).astype(np.uint8)
            else:
                self.median_background = frame_bgr.copy()
            self.calibrating = False
            self.phase = 'ready'
            self.phase_entered_time = now
            print("✅ Calibrage terminé ! Le décor vide est verrouillé.")
        return remaining

    def _get_fast_smooth_mask(self, frame_bgr, w, h):
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        seg_res = self.mp_selfie.process(small_rgb)
        raw_mask = seg_res.segmentation_mask.astype(np.float32)
        blurred_low = cv2.GaussianBlur(raw_mask, (7, 7), 1.4)
        stretched_low = np.clip((blurred_low - 0.12) / 0.68, 0.0, 1.0)
        smooth_mask = cv2.resize(stretched_low, (w, h), interpolation=cv2.INTER_LINEAR)
        return smooth_mask

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

    def trigger_action(self, mode, current_frame_bgr=None):
        now = time.time()
        self.last_trigger_time = now
        self.candidate_mode = None
        self.pose_start_time = None
        self.gesture_armed = False

        if self.phase == 'ready':
            self.phase = 'vanishing'
            self.active_mode = mode
            self.phase_start_time = now
            self.phase_entered_time = now
            if current_frame_bgr is not None:
                h, w = current_frame_bgr.shape[:2]
                self.cached_mask = self._get_fast_smooth_mask(current_frame_bgr, w, h)
                self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
                self.last_mask_time = now
                if mode == 'naruto':
                    self.saved_naruto_anchor = self.compute_substitution_anchor(self.cached_mask, w, h)
            print(f"✨ Disparition déclenchée : Mode {mode.upper()} !")
            self.play_sound(mode)
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
            self.play_sound('return')

    def compute_substitution_anchor(self, mask, w, h):
        sub = mask[::4, ::4]
        ay, ax = np.where(sub > 0.40)
        if len(ax) < 8:
            return (w // 2, h // 2, int(h * 0.57))
        min_x = int(np.min(ax)) * 4
        max_x = int(np.max(ax)) * 4
        min_y = int(np.min(ay)) * 4
        max_y = int(np.max(ay)) * 4

        sil_w = max_x - min_x
        sil_h = max_y - min_y
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
            axes = (puff, max(2, int(puff * 0.68)))
            cv2.ellipse(overlay, (px, py), axes, rot_deg, 0, 360, color_bgr, -1)

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
        x1 = lx - target_w // 2
        y1 = ly - target_h // 2
        x2 = x1 + target_w
        y2 = y1 + target_h

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = target_w - max(0, x2 - w_out)
        src_y2 = target_h - max(0, y2 - h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(w_out, x2)
        dst_y2 = min(h, y2)

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

    def process_frame(self, frame_bgr):
        now = time.time()
        h, w = frame_bgr.shape[:2]

        # 1. Calibration Phase (100% clean video feed)
        if self.calibrating:
            self.update_calibration(frame_bgr)
            return frame_bgr

        # 2. Asynchronous Gesture Detection in 'ready' or 'vanished' phases
        if self.phase in ('ready', 'vanished'):
            triggered_mode = None
            with self._gesture_lock:
                if self._pending_triggered_mode is not None:
                    triggered_mode = self._pending_triggered_mode
                    self._pending_triggered_mode = None
                elif self._gesture_frame is None:
                    self._gesture_frame = frame_bgr

            if triggered_mode is not None:
                self.trigger_action(triggered_mode, frame_bgr)

        # 3. Handle Vanish Phases
        bg = self.median_background if self.median_background is not None else frame_bgr
        elapsed = now - self.phase_start_time

        if self.phase == 'ready':
            self.saved_naruto_anchor = None
            return frame_bgr

        if self.phase == 'vanished':
            output = bg.copy()
            if self.active_mode == 'naruto':
                anchor = self.saved_naruto_anchor or (w // 2, h // 2, int(h * 0.57))
                self._render_naruto_log(output, anchor, h, 1.45)
            return output

        needs_fresh_mask = False
        if self.cached_mask is None or self.cached_bbox is None:
            needs_fresh_mask = True
        elif self.active_mode == 'normal' and (now - self.last_mask_time) >= 0.045:
            needs_fresh_mask = True
        elif self.active_mode == 'naruto' and elapsed < 0.13 and (now - self.last_mask_time) >= 0.045:
            needs_fresh_mask = True

        if needs_fresh_mask:
            self.cached_mask = self._get_fast_smooth_mask(frame_bgr, w, h)
            self.cached_bbox = self._compute_bbox_from_mask(self.cached_mask, w, h)
            self.last_mask_time = now

        smooth_mask = self.cached_mask
        bbox = self.cached_bbox

        if self.active_mode == 'naruto' and self.saved_naruto_anchor is None:
            self.saved_naruto_anchor = self.compute_substitution_anchor(smooth_mask, w, h)

        if self.phase == 'vanishing':
            duration = self.mode_durations.get(self.active_mode, 1.15)
            progress = min(1.0, elapsed / duration)

            if self.active_mode == 'goku':
                output = self.render_goku_shunkan_ido(frame_bgr, bg, smooth_mask, bbox, progress, int(now * 1000))
                if progress >= 0.94:
                    self.phase = 'vanished'
                    self.phase_entered_time = now
                return output

            elif self.active_mode == 'naruto':
                anchor = self.saved_naruto_anchor or (w // 2, h // 2, int(h * 0.57))
                if elapsed < 0.13:
                    survival = max(0.0, 1.0 - elapsed / 0.13)
                    output = self._blend_person_roi(frame_bgr, bg, smooth_mask, bbox, survival)
                else:
                    output = bg.copy()

                self._render_naruto_smoke_layer(output, anchor, elapsed, foreground=False)
                self._render_naruto_log(output, anchor, h, elapsed)
                self._render_naruto_smoke_layer(output, anchor, elapsed, foreground=True)

                if elapsed >= duration:
                    self.phase = 'vanished'
                    self.phase_entered_time = now
                return output

            else:
                survival = max(0.0, 1.0 - progress)
                output = self._blend_person_roi(frame_bgr, bg, smooth_mask, bbox, survival)
                if progress >= 1.0:
                    self.phase = 'vanished'
                    self.phase_entered_time = now
                return output

        elif self.phase == 'reappearing':
            reappear_duration = 0.50
            progress = min(1.0, elapsed / reappear_duration)

            if self.active_mode == 'goku':
                output = self.render_goku_shunkan_ido(frame_bgr, bg, smooth_mask, bbox, 1.0 - progress, int(now * 1000))
                if progress >= 1.0:
                    self.phase = 'ready'
                    self.phase_entered_time = now
                    self.active_mode = None
                return output

            elif self.active_mode == 'naruto':
                anchor = self.saved_naruto_anchor or (w // 2, h // 2, int(h * 0.57))
                appear_ratio = min(1.0, max(0.0, (progress - 0.14) / 0.35))
                output = self._blend_person_roi(frame_bgr, bg, smooth_mask, bbox, appear_ratio)

                smoke_age = progress * 1.15
                self._render_naruto_smoke_layer(output, anchor, smoke_age, foreground=False)
                self._render_naruto_smoke_layer(output, anchor, smoke_age, foreground=True)

                if progress >= 1.0:
                    self.phase = 'ready'
                    self.phase_entered_time = now
                    self.active_mode = None
                    self.saved_naruto_anchor = None
                return output

            else:
                output = self._blend_person_roi(frame_bgr, bg, smooth_mask, bbox, progress)
                if progress >= 1.0:
                    self.phase = 'ready'
                    self.phase_entered_time = now
                    self.active_mode = None
                return output

        return frame_bgr


def negotiate_native_camera_resolution(cap, requested_w=None, requested_h=None):
    """
    Adapts automatically to whatever resolution the user's webcam hardware offers:
    - Never forces software upscaling or an unsupported resolution.
    - If the user passed --width / --height explicitly, requests that resolution.
    - Otherwise, gently asks the webcam driver for its highest native 16:9 mode
      (the driver automatically clamps to 1080p, 720p, or 480p depending on the hardware)
      and reads a real hardware frame (`frame.shape[:2]`) to get the exact pixel dimensions.
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if requested_w is not None and requested_h is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_h)
    else:
        # Ask the driver for up to 1080p so OpenCV doesn't stay stuck in 640x480 default mode;
        # the camera driver automatically clamps to the highest resolution it natively supports.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, sample_frame = cap.read()
    if not ret or sample_frame is None:
        # Fallback if the camera rejected the resolution hint
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
    if c.isOpened():
        return c
    return None


def main():
    parser = argparse.ArgumentParser(description="VanishCam Pro - Real-Time Anime & VFX Webcam")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Optional custom width (default: Auto native camera resolution)")
    parser.add_argument("--height", type=int, default=None, help="Optional custom height (default: Auto native camera resolution)")
    parser.add_argument("--fps", type=int, default=30, help="Target output frame rate")
    args = parser.parse_args()

    print("=" * 60)
    print("🎥 VANISHCAM PRO — Real-Time Anime & VFX Virtual Camera")
    print("=" * 60)

    cap = None
    active_cam_idx = args.cam
    candidate_indices = [args.cam]
    for i in (0, 1, 2):
        if i not in candidate_indices:
            candidate_indices.append(i)

    for idx in candidate_indices:
        c = open_webcam_by_index(idx)
        if c is not None:
            cap = c
            active_cam_idx = idx
            print(f"📹 Webcam connectée (index {idx})")
            break

    if cap is None or not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        if sys.platform == 'darwin':
            print("💡 Sur macOS, vérifiez que votre Terminal est autorisé à utiliser la caméra :")
            print("   Réglages Système > Confidentialité et sécurité > Appareil photo")
        sys.exit(1)

    actual_w, actual_h = negotiate_native_camera_resolution(cap, args.width, args.height)
    print(f"✨ Résolution native détectée sur votre webcam : {actual_w}x{actual_h}")
    engine = VanishCamEngine(active_cam_idx, actual_w, actual_h, args.fps)

    fast_cam = ZeroLagCamera(cap)

    vcam = None
    if HAS_VIRTUAL_CAM:
        try:
            vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"✅ Caméra virtuelle active : {vcam.device} ({actual_w}x{actual_h})")
            print("👉 Choisissez 'OBS Virtual Camera' dans Discord / Teams / Zoom / Meet !")
        except Exception:
            print("ℹ️  Mode Fenêtre OpenCV actif (capturez la fenêtre 'VanishCam Pro' dans OBS Studio)")

    print("\nCommandes Clavier & 3 Gestes Actifs :")
    print("  [Espace] : Recalibrer le décor vide (4 secondes)")
    print("  [1]      : Disparition Fondu Normal (Geste : Signe V)")
    print("  [2]      : Substitution Naruto (Geste : Mains jointes / Mudra)")
    print("  [3]      : Téléportation Goku (Geste : 2 doigts sur le front)")
    print("  [C]      : Changer de caméra (si OBS Virtual Camera s'affiche au lieu de votre webcam)")
    print("  [R]      : Réapparaître immédiatement")
    print("  [Q]      : Quitter")
    print("-" * 60)

    window_name = "VanishCam Pro"
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
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                engine.start_calibration()
            elif key == ord('1'):
                engine.trigger_action('normal', frame)
            elif key == ord('2'):
                engine.trigger_action('naruto', frame)
            elif key == ord('3'):
                engine.trigger_action('goku', frame)
            elif key == ord('c') or key == ord('C'):
                for offset in (1, 2, 3):
                    next_idx = (active_cam_idx + offset) % 4
                    new_cap = open_webcam_by_index(next_idx)
                    if new_cap is not None:
                        fast_cam.release()
                        active_cam_idx = next_idx
                        actual_w, actual_h = negotiate_native_camera_resolution(new_cap, args.width, args.height)
                        fast_cam = ZeroLagCamera(new_cap)
                        print(f"🔄 Basculement vers la caméra index {active_cam_idx} ({actual_w}x{actual_h})")
                        engine.start_calibration()
                        break
            elif key == ord('r'):
                if engine.phase == 'vanished':
                    engine.trigger_action('return', frame)

    finally:
        engine.close()
        fast_cam.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        print("👋 VanishCam Pro arrêté.")


if __name__ == "__main__":
    main()
