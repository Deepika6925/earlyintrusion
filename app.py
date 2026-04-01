import cv2
import json
import streamlit as st
import tempfile
import os
import numpy as np

from ultralytics import YOLO
from deepface import DeepFace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 🔥 Fix warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- CONFIG ----------------
with open("model_config.json") as f:
    config = json.load(f)

threshold = config["threshold"]
emotion_weight = config["emotion_weight"]
behaviour_weight = config["behaviour_weight"]

# ---------------- MODELS ----------------
yolo_model = YOLO("yolov8n.pt")

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options
)

pose_detector = vision.PoseLandmarker.create_from_options(options)

# ---------------- SCORING ----------------
def emotion_score(emotion):

    weights = {
        "angry":0.9,
        "fear":0.8,
        "sad":0.6,
        "surprise":0.6,
        "neutral":0.2,
        "happy":0.1
    }

    return weights.get(emotion,0.2)

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

    return min(score,1.0)

def suspicious_score(emotion, landmarks):

    e = emotion_score(emotion)
    b = behaviour_score(landmarks)

    score = (emotion_weight * e) + (behaviour_weight * b)

    return score

# ---------------- VIDEO PROCESS ----------------
def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()  # Streamlit display

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = yolo_model(frame, classes=[0])

        for r in results:

            boxes = r.boxes.xyxy

            for box in boxes:

                x1,y1,x2,y2 = map(int,box)

                person = frame[y1:y2,x1:x2]

                if person.size == 0:
                    continue

                try:
                    result = DeepFace.analyze(
                        person,
                        actions=['emotion'],
                        enforce_detection=False
                    )

                    emotion = result[0]['dominant_emotion']

                except:
                    emotion = "neutral"

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

                color = (0,0,255) if label == "Suspicious" else (0,255,0)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,color,2)

        # 🔥 Show in Streamlit instead of cv2.imshow
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

# ---------------- UI ----------------
st.title("Suspicious Activity Detection System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if uploaded_file is not None:

    st.info("Processing video... please wait ⏳")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    analyze_video(tfile.name)

    st.success("Processing completed ✅")
