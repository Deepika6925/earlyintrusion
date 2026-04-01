import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import json
import gradio as gr
import numpy as np

from ultralytics import YOLO
from deepface import DeepFace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ---------------- #
with open("model_config.json") as f:
    config = json.load(f)

threshold = config["threshold"]
emotion_weight = config["emotion_weight"]
behaviour_weight = config["behaviour_weight"]

# ---------------- MODELS ---------------- #
yolo_model = YOLO("yolov8n.pt")

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# ---------------- LOGIC ---------------- #
def emotion_score(emotion):
    weights = {
        "angry": 0.9,
        "fear": 0.8,
        "sad": 0.6,
        "surprise": 0.6,
        "neutral": 0.2,
        "happy": 0.1
    }
    return weights.get(emotion, 0.2)

def behaviour_score(landmarks):
    if landmarks is None:
        return 0.3

    nose = landmarks[0].y
    left_wrist = landmarks[15].y
    right_wrist = landmarks[16].y

    score = 0.3

    if left_wrist < nose:
        score += 0.3
    if right_wrist < nose:
        score += 0.3

    return min(score, 1.0)

def suspicious_score(emotion, landmarks):
    e = emotion_score(emotion)
    b = behaviour_score(landmarks)
    return (emotion_weight * e) + (behaviour_weight * b)

# ---------------- MAIN ---------------- #
def analyze_video(video):

    cap = cv2.VideoCapture(video)
    frame_count = 0
    output_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        results = yolo_model(frame, classes=[0])

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                person = frame[y1:y2, x1:x2]

                if person.size == 0:
                    continue

                # Emotion
                try:
                    result = DeepFace.analyze(
                        person,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    emotion = result[0]['dominant_emotion']
                except:
                    emotion = "neutral"

                # Pose (Tasks API)
                rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                pose_result = pose_detector.detect(mp_image)

                landmarks = None
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks[0]

                score = suspicious_score(emotion, landmarks)
                label = "Suspicious" if score > threshold else "Normal"

                color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({emotion})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        output_frame = frame

    cap.release()

    if output_frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    return output_frame

# ---------------- UI ---------------- #
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(),
    outputs="image",
    title="Suspicious Activity Detection System"
)

iface.launch(server_name="0.0.0.0", server_port=10000)
