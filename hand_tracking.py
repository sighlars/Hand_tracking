from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

CAM_INDEX = 0

# 640 x 480 gives better real time responsiveness on weaker
# laptops than 1280 x 720.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.8

# Main smoothing value.
# Smaller values are smoother but introduce more latency.
SMOOTHING_ALPHA = 0.30

# Maximum movement allowed between consecutive frames before
# treating the movement as tracking noise.
MAX_JUMP_DISTANCE = 100

# Speed at which adaptive smoothing becomes more responsive.
FAST_MOVEMENT_SPEED = 35.0

COLOR_BG = (18, 18, 18)
COLOR_ACCENT = (0, 210, 255)
COLOR_TEXT = (235, 235, 235)
COLOR_LEFT = (255, 140, 0)
COLOR_RIGHT = (0, 200, 120)
COLOR_PINCH = (60, 60, 255)

BRUSH_COLORS = [
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

DEFAULT_BRUSH_SIZE = 6
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 40


# ============================================================
# LANDMARK INDICES
# ============================================================

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8


# ============================================================
# FPS COUNTER
# ============================================================

@dataclass
class FPSCounter:

    _prev_time: float = field(default_factory=time.time)
    _fps: float = 0.0
    smoothing: float = 0.9

    def tick(self) -> float:

        now = time.time()
        dt = now - self._prev_time

        self._prev_time = now

        if dt > 0:

            instant_fps = 1.0 / dt

            self._fps = (
                self._fps * self.smoothing
                + instant_fps * (1.0 - self.smoothing)
            )

        return self._fps


# ============================================================
# HAND READING
# ============================================================

@dataclass
class HandReading:

    label: str
    confidence: float
    landmarks_px: List[Tuple[int, int]]
    fingers_up: List[bool]

    @property
    def total_fingers(self) -> int:

        return sum(self.fingers_up)

    @property
    def index_tip(self) -> Tuple[int, int]:

        return self.landmarks_px[INDEX_TIP_IDX]

    @property
    def thumb_tip(self) -> Tuple[int, int]:

        return self.landmarks_px[THUMB_TIP_IDX]

    @property
    def is_pointer_gesture(self) -> bool:

        return self.fingers_up == [
            False,
            True,
            False,
            False,
            False,
        ]

    @property
    def is_open_palm(self) -> bool:

        return all(self.fingers_up)

    def pinch_distance(self) -> float:

        x1, y1 = self.thumb_tip
        x2, y2 = self.index_tip

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )


# ============================================================
# HAND TRACKER
# ============================================================

class HandTracker:

    def __init__(
        self,
        max_hands: int = MAX_HANDS,
        detection_confidence: float = DETECTION_CONFIDENCE,
        tracking_confidence: float = TRACKING_CONFIDENCE,
    ) -> None:

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def process(self, frame_bgr) -> List[HandReading]:

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        frame_rgb.flags.writeable = False

        results = self._hands.process(frame_rgb)

        frame_rgb.flags.writeable = True

        readings: List[HandReading] = []

        if not results.multi_hand_landmarks:

            return readings

        h, w = frame_bgr.shape[:2]

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):

            label = handedness.classification[0].label

            confidence = handedness.classification[0].score

            points_px = [
                (
                    int(lm.x * w),
                    int(lm.y * h),
                )
                for lm in hand_landmarks.landmark
            ]

            fingers_up = self._fingers_up(
                hand_landmarks.landmark,
                label,
            )

            readings.append(
                HandReading(
                    label=label,
                    confidence=confidence,
                    landmarks_px=points_px,
                    fingers_up=fingers_up,
                )
            )

            self._draw_landmarks(
                frame_bgr,
                hand_landmarks,
            )

        return readings

    def _draw_landmarks(
        self,
        frame_bgr,
        hand_landmarks,
    ) -> None:

        self._mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style(),
        )

    @staticmethod
    def _fingers_up(
        landmarks,
        label: str,
    ) -> List[bool]:

        fingers = []

        thumb_tip_x = landmarks[FINGER_TIPS[0]].x
        thumb_pip_x = landmarks[FINGER_PIPS[0]].x

        if label == "Right":

            fingers.append(
                thumb_tip_x < thumb_pip_x
            )

        else:

            fingers.append(
                thumb_tip_x > thumb_pip_x
            )

        for tip_idx, pip_idx in zip(
            FINGER_TIPS[1:],
            FINGER_PIPS[1:],
        ):

            fingers.append(
                landmarks[tip_idx].y
                < landmarks[pip_idx].y
            )

        return fingers

    def close(self) -> None:

        self._hands.close()


# ============================================================
# DRAWING CANVAS
# ============================================================

class DrawingCanvas:

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:

        self.canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        # Previous point actually used for drawing.
        self._last_points: Dict[
            str,
            Optional[Tuple[int, int]]
        ] = {}

        # Previous smoothed point.
        self._smooth_points: Dict[
            str,
            Optional[Tuple[int, int]]
        ] = {}

        self._previous_raw_points: Dict[
            str,
            Optional[Tuple[int, int]]
        ] = {}

        self._speeds: Dict[str, float] = {}

        self.brush_color = BRUSH_COLORS[0]

        self.brush_size = DEFAULT_BRUSH_SIZE

    # --------------------------------------------------------
    # SMOOTHING
    # --------------------------------------------------------

    def smooth_point(
        self,
        hand_key: str,
        point: Tuple[int, int],
    ) -> Tuple[int, int]:

        previous = self._smooth_points.get(hand_key)

        previous_raw = self._previous_raw_points.get(
            hand_key
        )

        # First point
        if previous is None:

            self._smooth_points[hand_key] = point

            self._previous_raw_points[hand_key] = point

            self._speeds[hand_key] = 0.0

            return point

        # ----------------------------------------------------
        # Calculate raw movement speed
        # ----------------------------------------------------

        if previous_raw is not None:

            dx = point[0] - previous_raw[0]
            dy = point[1] - previous_raw[1]

            speed = math.hypot(dx, dy)

        else:

            speed = 0.0

        self._speeds[hand_key] = speed

        self._previous_raw_points[hand_key] = point

        # ----------------------------------------------------
        # Reject impossible tracking jumps
        # ----------------------------------------------------

        raw_jump = math.hypot(
            point[0] - previous[0],
            point[1] - previous[1],
        )

        if raw_jump > MAX_JUMP_DISTANCE:

            return previous

        # ----------------------------------------------------
        # Adaptive smoothing
        # ----------------------------------------------------
        #
        # Slow movement:
        #       stronger smoothing
        #
        # Fast movement:
        #       weaker smoothing
        #
        # This prevents the cursor from feeling sluggish.
        # ----------------------------------------------------

        if speed < FAST_MOVEMENT_SPEED:

            alpha = SMOOTHING_ALPHA

        else:

            alpha = min(
                0.75,
                SMOOTHING_ALPHA
                + (speed / 150.0),
            )

        x = int(
            previous[0]
            + alpha * (point[0] - previous[0])
        )

        y = int(
            previous[1]
            + alpha * (point[1] - previous[1])
        )

        smoothed = (x, y)

        self._smooth_points[hand_key] = smoothed

        return smoothed

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def stroke_to(
        self,
        hand_key: str,
        point: Tuple[int, int],
    ) -> None:

        last = self._last_points.get(hand_key)

        if last is not None:

            distance = math.hypot(
                point[0] - last[0],
                point[1] - last[1],
            )

            # Prevent accidental giant lines.
            if distance <= MAX_JUMP_DISTANCE:

                cv2.line(
                    self.canvas,
                    last,
                    point,
                    self.brush_color,
                    self.brush_size,
                    cv2.LINE_AA,
                )

        else:

            cv2.circle(
                self.canvas,
                point,
                max(1, self.brush_size // 2),
                self.brush_color,
                -1,
                cv2.LINE_AA,
            )

        self._last_points[hand_key] = point

    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    def release(
        self,
        hand_key: str,
    ) -> None:

        self._last_points[hand_key] = None

    # --------------------------------------------------------
    # ACTIVE HANDS
    # --------------------------------------------------------

    def active_keys(self):

        return list(
            self._last_points.keys()
        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:

        self.canvas[:] = 0

        self._last_points.clear()

        self._smooth_points.clear()

        self._previous_raw_points.clear()

        self._speeds.clear()

    # --------------------------------------------------------
    # RESET HAND
    # --------------------------------------------------------

    def reset_hand(
        self,
        hand_key: str,
    ) -> None:

        self._last_points[hand_key] = None

        self._smooth_points[hand_key] = None

        self._previous_raw_points[hand_key] = None

        self._speeds[hand_key] = 0.0

    # --------------------------------------------------------
    # COMPOSITE
    # --------------------------------------------------------

    def composite_onto(
        self,
        frame,
    ) -> None:

        mask = (
            cv2.cvtColor(
                self.canvas,
                cv2.COLOR_BGR2GRAY,
            )
            > 0
        )

        if not np.any(mask):

            return

        blended = cv2.addWeighted(
            frame,
            0.25,
            self.canvas,
            0.9,
            0,
        )

        frame[mask] = blended[mask]

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    def cycle_color(
        self,
        index: int,
    ) -> None:

        if 0 <= index < len(BRUSH_COLORS):

            self.brush_color = BRUSH_COLORS[index]

    # --------------------------------------------------------
    # BRUSH SIZE
    # --------------------------------------------------------

    def resize_brush(
        self,
        delta: int,
    ) -> None:

        self.brush_size = max(
            MIN_BRUSH_SIZE,
            min(
                MAX_BRUSH_SIZE,
                self.brush_size + delta,
            ),
        )


# ============================================================
# HUD
# ============================================================

def draw_hud(
    frame,
    fps: float,
    readings: List[HandReading],
    canvas: DrawingCanvas,
) -> None:

    h, w = frame.shape[:2]

    panel_h = 100 + 34 * max(
        len(readings),
        1,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (380, panel_h),
        COLOR_BG,
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_ACCENT,
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (320, 27),
        canvas.brush_size,
        canvas.brush_color,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (320, 27),
        canvas.brush_size,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = 66

    if not readings:

        cv2.putText(
            frame,
            "No hand detected",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        y += 34

    for reading in readings:

        if reading.label == "Left":

            color = COLOR_LEFT

        else:

            color = COLOR_RIGHT

        if reading.is_pointer_gesture:

            gesture = "DRAWING"

        elif reading.is_open_palm:

            gesture = "CLEAR"

        else:

            gesture = ""

        text = (
            f"{reading.label}: "
            f"{reading.total_fingers} finger(s) "
            f"{gesture}"
        )

        cv2.putText(
            frame,
            text,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 34

    if readings:

        dist = readings[0].pinch_distance()

        cv2.putText(
            frame,
            f"Pinch dist: {dist:5.0f}px",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_PINCH,
            2,
            cv2.LINE_AA,
        )


# ============================================================
# PINCH INDICATOR
# ============================================================

def draw_pinch_indicator(
    frame,
    reading: HandReading,
) -> None:

    p1 = reading.thumb_tip
    p2 = reading.index_tip

    mid = (
        (p1[0] + p2[0]) // 2,
        (p1[1] + p2[1]) // 2,
    )

    cv2.line(
        frame,
        p1,
        p2,
        COLOR_PINCH,
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        p1,
        6,
        COLOR_PINCH,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        p2,
        6,
        COLOR_PINCH,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        mid,
        4,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )


# ============================================================
# FOOTER
# ============================================================

def draw_footer(frame) -> None:

    h, w = frame.shape[:2]

    text = (
        "q: quit   "
        "s: screenshot   "
        "c: clear   "
        "1-4: color   "
        "+/-: brush size"
    )

    cv2.putText(
        frame,
        text,
        (16, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    cap = cv2.VideoCapture(
        CAM_INDEX,
        cv2.CAP_DSHOW,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    # Request a reasonable webcam FPS.
    cap.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open webcam at index {CAM_INDEX}. "
            "Check that it is connected and not in use by another app."
        )

    actual_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ) or FRAME_WIDTH

    actual_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ) or FRAME_HEIGHT

    tracker = HandTracker()

    canvas = DrawingCanvas(
        actual_w,
        actual_h,
    )

    fps_counter = FPSCounter()

    screenshot_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "screenshots",
    )

    window_name = "Hand Tracking and Air Drawing"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    print()
    print("Hand tracking started.")
    print()
    print("Point with only your index finger to draw.")
    print("Open your palm to clear the canvas.")
    print("Press q to quit.")
    print("Press s to save a screenshot.")
    print("Press c to clear.")
    print("Press 1 to 4 to change brush color.")
    print("Press + or = to increase brush size.")
    print("Press - or _ to decrease brush size.")
    print()

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "Failed to read frame from webcam."
                )

                break

            # Mirror the camera.
            frame = cv2.flip(
                frame,
                1,
            )

            # Process hand tracking.
            readings = tracker.process(
                frame
            )

            fps = fps_counter.tick()

            active_hand_keys = set()

            should_clear = False

            # ------------------------------------------------
            # Process every detected hand
            # ------------------------------------------------

            for i, reading in enumerate(
                readings
            ):

                hand_key = (
                    f"{reading.label}_{i}"
                )

                active_hand_keys.add(
                    hand_key
                )

                # --------------------------------------------
                # Drawing gesture
                # --------------------------------------------

                if reading.is_pointer_gesture:

                    # Smooth the raw fingertip.
                    smooth_point = (
                        canvas.smooth_point(
                            hand_key,
                            reading.index_tip,
                        )
                    )

                    # Draw using the smoothed point.
                    canvas.stroke_to(
                        hand_key,
                        smooth_point,
                    )

                    # Draw a cursor around the smoothed
                    # fingertip instead of the noisy raw point.
                    cv2.circle(
                        frame,
                        smooth_point,
                        canvas.brush_size + 5,
                        canvas.brush_color,
                        2,
                        cv2.LINE_AA,
                    )

                    # Small center point.
                    cv2.circle(
                        frame,
                        smooth_point,
                        3,
                        canvas.brush_color,
                        -1,
                        cv2.LINE_AA,
                    )

                else:

                    canvas.release(
                        hand_key
                    )

                # --------------------------------------------
                # Open palm
                # --------------------------------------------

                if reading.is_open_palm:

                    should_clear = True

                # --------------------------------------------
                # Pinch
                # --------------------------------------------

                draw_pinch_indicator(
                    frame,
                    reading,
                )

            # ------------------------------------------------
            # Release hands that disappeared
            # ------------------------------------------------

            for key in canvas.active_keys():

                if key not in active_hand_keys:

                    canvas.reset_hand(
                        key
                    )

            # ------------------------------------------------
            # Clear canvas
            # ------------------------------------------------

            if should_clear:

                canvas.clear()

            # ------------------------------------------------
            # Draw persistent canvas
            # ------------------------------------------------

            canvas.composite_onto(
                frame
            )

            # ------------------------------------------------
            # UI
            # ------------------------------------------------

            draw_hud(
                frame,
                fps,
                readings,
                canvas,
            )

            draw_footer(
                frame
            )

            # ------------------------------------------------
            # Show
            # ------------------------------------------------

            cv2.imshow(
                window_name,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):

                break

            # Screenshot
            elif key == ord("s"):

                os.makedirs(
                    screenshot_dir,
                    exist_ok=True,
                )

                filename = os.path.join(
                    screenshot_dir,
                    f"capture_{int(time.time())}.png",
                )

                cv2.imwrite(
                    filename,
                    frame,
                )

                print(
                    f"Saved screenshot: {filename}"
                )

            # Clear
            elif key == ord("c"):

                canvas.clear()

            # Increase brush
            elif key in (
                ord("+"),
                ord("="),
            ):

                canvas.resize_brush(
                    2
                )

            # Decrease brush
            elif key in (
                ord("-"),
                ord("_"),
            ):

                canvas.resize_brush(
                    -2
                )

            # Colors
            elif key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
            ):

                canvas.cycle_color(
                    key - ord("1")
                )

    finally:

        tracker.close()

        cap.release()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()