import streamlit as st
import cv2
import numpy as np
import json
from ultralytics import YOLO
import tempfile

# Load config
with open("model_config.json") as f:
    config = json.load(f)

threshold = config["threshold"]
emotion_weight = config["emotion_weight"]
behaviour_weight = config["behaviour_weight"]

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# ------------------ Emotion Approximation ------------------
def get_emotion(person_img):
    gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    if mean_intensity < 80:
        return "angry"
    elif mean_intensity < 120:
        return "sad"
    elif mean_intensity < 160:
        return "neutral"
    else:
        return "happy"

# ------------------ Behaviour Approximation ------------------
def behaviour_score(box):
    x1, y1, x2, y2 = box

    height = y2 - y1
    width = x2 - x1

    score = 0.3

    # heuristic: if bounding box is tall → maybe raised hands
    if height > width * 1.2:
        score += 0.3

    # random slight variation for realism
    score += np.random.uniform(0, 0.2)

    return min(score, 1.0)

# ------------------ Emotion Score ------------------
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

# ------------------ Final Score ------------------
def suspicious_score(emotion, behaviour):
    e = emotion_score(emotion)
    b = behaviour

    return (emotion_weight * e) + (behaviour_weight * b)

# ------------------ Video Processing ------------------
def process_video(video_file):

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, classes=[0])

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                person = frame[y1:y2, x1:x2]
                if person.size == 0:
                    continue

                # Emotion (approx)
                emotion = get_emotion(person)

                # Behaviour (approx)
                behaviour = behaviour_score(box)

                score = suspicious_score(emotion, behaviour)

                label = "Suspicious" if score > threshold else "Normal"
                color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

# ------------------ Streamlit UI ------------------
st.title("Suspicious Activity Detection System")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file is not None:
    process_video(video_file)
