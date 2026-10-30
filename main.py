
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "face_landmarker.task"
ANOMALY_DURATION_SECONDS = 2.0

# Initial calibration values (configurable).
# These are not specified by the task reference.
HORIZONTAL_TOLERANCE = 0.15
VERTICAL_TOLERANCE = 0.12

LOG_FILE = "gaze_anomalies.log"


# --------------------------------------------------
# MEDIAPIPE FACE LANDMARKER
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# --------------------------------------------------
# EYE LANDMARK INDICES
# --------------------------------------------------

# Eye corner anchors and iris landmarks.
EYES = {
    "Eye 1": {
        "corners": (33, 133),
        "iris": (468, 469, 470, 471, 472)
    },
    "Eye 2": {
        "corners": (362, 263),
        "iris": (473, 474, 475, 476, 477)
    }
}


# --------------------------------------------------
# ANOMALY LOGGING
# --------------------------------------------------

def log_anomaly(horizontal_shift, vertical_shift, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"[{timestamp}] OFF-CENTER GAZE DETECTED | "
        f"Horizontal Shift: {horizontal_shift:.3f} | "
        f"Vertical Shift: {vertical_shift:.3f} | "
        f"Duration: {duration:.2f} seconds\n"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(message)

    print(message.strip())


# --------------------------------------------------
# CUSTOM GAZE VECTOR CALCULATION
# --------------------------------------------------

def calculate_eye_vector(landmarks, eye_data, frame_width, frame_height):

    left_index, right_index = eye_data["corners"]

    left_corner = landmarks[left_index]
    right_corner = landmarks[right_index]

    iris_points = [
        landmarks[index]
        for index in eye_data["iris"]
    ]

    # Convert normalized coordinates to pixel coordinates.
    left_x = left_corner.x * frame_width
    left_y = left_corner.y * frame_height

    right_x = right_corner.x * frame_width
    right_y = right_corner.y * frame_height

    iris_x = np.mean([
        point.x * frame_width for point in iris_points
    ])

    iris_y = np.mean([
        point.y * frame_height for point in iris_points
    ])

    # Calculate eye center using corner anchors.
    eye_center_x = (left_x + right_x) / 2
    eye_center_y = (left_y + right_y) / 2

    # Calculate eye width.
    eye_width = np.sqrt(
        (right_x - left_x) ** 2 +
        (right_y - left_y) ** 2
    )

    if eye_width == 0:
        return None

    # Custom normalized gaze displacement equations.
    horizontal_shift = (iris_x - eye_center_x) / eye_width
    vertical_shift = (iris_y - eye_center_y) / eye_width

    return horizontal_shift, vertical_shift, (int(iris_x), int(iris_y))


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        print(f"Model file not found: {MODEL_PATH}")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Unable to access webcam.")
        return

    anomaly_start_time = None
    anomaly_logged = False
    frame_timestamp = 0
    start_time = time.monotonic()

    print("Day 24 - Computer Vision Gaze Math")
    print("Webcam started successfully.")
    print("Press Q or ESC to exit.")

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)

                frame_height, frame_width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                current_timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                frame_timestamp = max(
                    frame_timestamp + 1,
                    current_timestamp
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    frame_timestamp
                )

                gaze_values = []

                if result.face_landmarks:

                    landmarks = result.face_landmarks[0]

                    for eye_name, eye_data in EYES.items():

                        vector = calculate_eye_vector(
                            landmarks,
                            eye_data,
                            frame_width,
                            frame_height
                        )

                        if vector is None:
                            continue

                        horizontal, vertical, iris_center = vector

                        gaze_values.append((horizontal, vertical))

                        cv2.circle(
                            frame,
                            iris_center,
                            4,
                            (0, 255, 0),
                            -1
                        )

                        cv2.putText(
                            frame,
                            eye_name,
                            (iris_center[0] + 8, iris_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1
                        )

                    if gaze_values:

                        # Average the two eye displacement vectors.
                        horizontal_shift = float(np.mean([
                            value[0] for value in gaze_values
                        ]))

                        vertical_shift = float(np.mean([
                            value[1] for value in gaze_values
                        ]))

                        # Check whether gaze is outside tolerance.
                        off_center = (
                            abs(horizontal_shift) > HORIZONTAL_TOLERANCE
                            or
                            abs(vertical_shift) > VERTICAL_TOLERANCE
                        )

                        if off_center:

                            if anomaly_start_time is None:
                                anomaly_start_time = time.monotonic()
                                anomaly_logged = False

                            duration = (
                                time.monotonic() - anomaly_start_time
                            )

                            if (
                                duration > ANOMALY_DURATION_SECONDS
                                and not anomaly_logged
                            ):

                                log_anomaly(
                                    horizontal_shift,
                                    vertical_shift,
                                    duration
                                )

                                anomaly_logged = True

                            status = "OFF-CENTER GAZE"

                            status_color = (0, 0, 255)

                        else:

                            anomaly_start_time = None
                            anomaly_logged = False

                            duration = 0
                            status = "CENTERED GAZE"
                            status_color = (0, 255, 0)

                        cv2.putText(
                            frame,
                            status,
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            status_color,
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Horizontal Shift: {horizontal_shift:.3f}",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Vertical Shift: {vertical_shift:.3f}",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Off-center Duration: {duration:.2f}s",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                else:

                    anomaly_start_time = None
                    anomaly_logged = False

                    cv2.putText(
                        frame,
                        "No Face Detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow(
                    "Day 24 - Computer Vision Gaze Math",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("Exit key detected.")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")
        print("Day 24 application closed.")


if __name__ == "__main__":
    main()