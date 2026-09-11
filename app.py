import cv2
import numpy as np
import serial
import time
import os
import math

from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose


# ============================================================
# SETTINGS
# ============================================================

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 115200

PALM_MODEL = "palm_detection_mediapipe_2023feb.onnx"
HAND_MODEL = "handpose_estimation_mediapipe_2023feb.onnx"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Smoothing:
# 0.0 = extremely smooth/slow
# 1.0 = no smoothing
SMOOTHING = 0.55

# Number of identical frames required before changing LED
STABLE_FRAMES = 4


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(PALM_MODEL):
    print(f"Missing model: {PALM_MODEL}")
    exit()

if not os.path.exists(HAND_MODEL):
    print(f"Missing model: {HAND_MODEL}")
    exit()


# ============================================================
# ESP32
# ============================================================

try:
    esp32 = serial.Serial(
        PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("ESP32 connected.")

except serial.SerialException as e:
    print("Could not connect to ESP32.")
    print(e)
    exit()


# ============================================================
# MODELS
# ============================================================

print("Loading hand detector...")

palm_detector = MPPalmDet(
    modelPath=PALM_MODEL,

    # Slightly lower threshold helps prevent missed hands
    nmsThreshold=0.3,
    scoreThreshold=0.55,

    backendId=cv2.dnn.DNN_BACKEND_OPENCV,
    targetId=cv2.dnn.DNN_TARGET_CPU
)

handpose_detector = MPHandPose(
    modelPath=HAND_MODEL,

    # 0.65 is a nice balance
    confThreshold=0.65,

    backendId=cv2.dnn.DNN_BACKEND_OPENCV,
    targetId=cv2.dnn.DNN_TARGET_CPU
)

print("Models loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera 💀")
    esp32.close()
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Try to reduce camera buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(hand):

    data = np.asarray(hand, dtype=np.float32).flatten()

    # MPHandPose output:
    #
    # 0:4      = bbox
    # 4:67     = 21 landmarks * 3
    # 67:130   = world landmarks
    # 130      = handedness
    # 131      = confidence

    if len(data) < 67:
        return None

    # THIS IS THE IMPORTANT FIX
    landmarks = data[4:67].reshape(21, 3)

    # Only x/y needed for drawing and finger counting
    points = landmarks[:, :2]

    return points


# ============================================================
# ANGLE FUNCTION
# ============================================================

def angle_between(a, b, c):

    """
    Calculates angle ABC.

    a ---- b ---- c

    180 degrees = finger is straight
    90 degrees  = finger is bent
    """

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# ============================================================
# FINGER COUNTING
# ============================================================

def count_fingers(points):

    """
    MediaPipe-style landmark indices:

             8   12  16  20
             |   |   |   |
             7   11  15  19
             |   |   |   |
             6   10  14  18
             |   |   |   |
         4   5   9   13  17
          \  |   |   |  /
           \ |   |   | /
              0

    0 = wrist
    """

    if points is None or len(points) != 21:
        return 0

    points = np.asarray(points, dtype=np.float32)

    count = 0


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index_angle = angle_between(
        points[5],
        points[6],
        points[8]
    )

    if index_angle > 155:
        count += 1


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle_angle = angle_between(
        points[9],
        points[10],
        points[12]
    )

    if middle_angle > 155:
        count += 1


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring_angle = angle_between(
        points[13],
        points[14],
        points[16]
    )

    if ring_angle > 155:
        count += 1


    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    pinky_angle = angle_between(
        points[17],
        points[18],
        points[20]
    )

    if pinky_angle > 155:
        count += 1


    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_angle = angle_between(
        points[1],
        points[2],
        points[4]
    )

    # Thumb also needs to be reasonably far from wrist
    wrist_to_thumb = np.linalg.norm(
        points[4] - points[0]
    )

    thumb_base_to_tip = np.linalg.norm(
        points[2] - points[4]
    )

    if thumb_angle > 145 and wrist_to_thumb > thumb_base_to_tip * 1.35:
        count += 1


    return count


# ============================================================
# LANDMARK SMOOTHING
# ============================================================

previous_points = None


def smooth_landmarks(points):

    global previous_points

    if points is None:
        return previous_points

    if previous_points is None:
        previous_points = points.copy()
        return points

    # Exponential moving average
    smoothed = (
        SMOOTHING * points
        +
        (1.0 - SMOOTHING) * previous_points
    )

    previous_points = smoothed

    return smoothed


# ============================================================
# SERIAL STATE
# ============================================================

last_sent_count = -1

candidate_count = -1
candidate_frames = 0


def send_to_esp32(count):

    global last_sent_count
    global candidate_count
    global candidate_frames

    # --------------------------------------------------------
    # Stability filter
    # --------------------------------------------------------

    if count != candidate_count:

        candidate_count = count
        candidate_frames = 1

        return

    else:

        candidate_frames += 1


    # Don't change LED until gesture is stable
    if candidate_frames < STABLE_FRAMES:
        return


    # Already sent
    if count == last_sent_count:
        return


    # --------------------------------------------------------
    # SEND COMMAND
    # --------------------------------------------------------

    if count == 1:

        esp32.write(b"1")
        print("☝️  1 finger -> LED 1")

    elif count == 2:

        esp32.write(b"2")
        print("✌️  2 fingers -> LED 2")

    elif count == 3:

        esp32.write(b"3")
        print("🤟  3 fingers -> LED 3")

    elif count == 4:

        esp32.write(b"4")
        print("🖖  4 fingers -> LED 4")

    else:

        esp32.write(b"0")
        print("0/5 fingers -> LEDs OFF")


    last_sent_count = count


# ============================================================
# FPS
# ============================================================

prev_time = time.time()
fps = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera said no 💀")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    finger_count = 0
    points = None


    # ========================================================
    # PALM DETECTION
    # ========================================================

    palms = palm_detector.infer(frame)


    if palms is not None and len(palms) > 0:

        # ----------------------------------------------------
        # Pick the largest detected palm
        # rather than blindly using palms[0]
        # ----------------------------------------------------

        def palm_area(p):

            p = np.asarray(p)

            if len(p) < 4:
                return 0

            x1, y1, x2, y2 = p[:4]

            return abs((x2 - x1) * (y2 - y1))


        palm = max(
            palms,
            key=palm_area
        )


        # ====================================================
        # HAND LANDMARKS
        # ====================================================

        hand = handpose_detector.infer(
            frame,
            palm
        )


        if hand is not None:

            points = extract_landmarks(hand)


            if points is not None:

                # ------------------------------------------------
                # SMOOTH LANDMARKS
                # ------------------------------------------------

                points = smooth_landmarks(points)


                # ------------------------------------------------
                # COUNT FINGERS
                # ------------------------------------------------

                finger_count = count_fingers(points)


                # =================================================
                # DRAW LANDMARKS
                # =================================================

                for i, point in enumerate(points):

                    x = int(point[0])
                    y = int(point[1])

                    # Don't draw insane coordinates
                    if (
                        0 <= x < frame.shape[1]
                        and
                        0 <= y < frame.shape[0]
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )


                # =================================================
                # DRAW SKELETON
                # =================================================

                for a, b in connections:

                    x1 = int(points[a][0])
                    y1 = int(points[a][1])

                    x2 = int(points[b][0])
                    y2 = int(points[b][1])


                    if (
                        0 <= x1 < frame.shape[1]
                        and
                        0 <= y1 < frame.shape[0]
                        and
                        0 <= x2 < frame.shape[1]
                        and
                        0 <= y2 < frame.shape[0]
                    ):

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 255),
                            2
                        )


    # ========================================================
    # SEND TO ESP32
    # ========================================================

    send_to_esp32(finger_count)


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    delta = current_time - prev_time

    if delta > 0:

        fps = 0.9 * fps + 0.1 * (1.0 / delta)

    prev_time = current_time


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "1-4 fingers = LEDs | Q = quit",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "ESP32 Hand Control",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
esp32.close()

print("Program stopped.")