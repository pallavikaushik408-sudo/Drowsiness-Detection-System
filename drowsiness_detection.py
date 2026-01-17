# =============================
# Drowsiness Detection System (Normalized Eye Detection)
# =============================
# Requirements:
# pip install ultralytics opencv-python pygame requests numpy
# Place an 'alarm.wav' file in the same folder.

import os
import requests
import cv2
import pygame
from ultralytics import YOLO
import numpy as np
from collections import deque

# ----------------------------
# Step 1: Download YOLO open/closed eyes model
# ----------------------------
model_path = "open_closed_eye_model.pt"
if not os.path.exists(model_path):
    print("🔽 Downloading open_closed_eye_model.pt ...")
    url = "https://huggingface.co/MichalMlodawski/open-closed-eye-detection/resolve/main/model.pt"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        exit()
else:
    print("✅ Model already exists locally.")

# ----------------------------
# Step 2: Load YOLO model
# ----------------------------
model = YOLO(model_path)
print("✅ Model loaded successfully!")

# ----------------------------
# Step 3: Initialize pygame for alarm & display
# ----------------------------
pygame.init()
pygame.mixer.quit()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

alarm_file = "alarm.wav"
if not os.path.exists(alarm_file):
    print(f"❌ Alarm file '{alarm_file}' not found!")
    exit()

alarm_sound = pygame.mixer.Sound(alarm_file)
alarm_sound.set_volume(1.0)
alarm_channel = pygame.mixer.Channel(0)

# Webcam setup
WEBCAM_WIDTH, WEBCAM_HEIGHT = 640, 480
screen = pygame.display.set_mode((WEBCAM_WIDTH, WEBCAM_HEIGHT))
pygame.display.set_caption("Drowsiness Detection")
font = pygame.font.SysFont("Arial", 40, bold=True)

# ----------------------------
# Step 4: Start webcam
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

# ----------------------------
# Detection parameters
# ----------------------------
SMOOTHING_FRAMES = 10
drowsy_queue = deque(maxlen=SMOOTHING_FRAMES)
EYE_CLOSED_THRESHOLD = 0.6  # Closed probability threshold

print("Press 'q' in the window to quit...")

# ----------------------------
# Step 5: Main loop
# ----------------------------
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image

    # ----------------------------
    # Eye detection with normalization
    # ----------------------------
    closed_confidence = 0
    open_confidence = 0

    results = model(frame)
    for obj in results:
        if len(obj.boxes) > 0:
            for cls, conf in zip(obj.boxes.cls, obj.boxes.conf):
                label = model.names[int(cls)].lower()
                if label == "closed":
                    closed_confidence = max(closed_confidence, conf)
                elif label == "open":
                    open_confidence = max(open_confidence, conf)

    # Normalize closed probability
    total_conf = closed_confidence + open_confidence + 1e-6
    closed_prob = closed_confidence / total_conf
    eyes_closed = closed_prob >= EYE_CLOSED_THRESHOLD

    # ----------------------------
    # Smoothing over frames
    # ----------------------------
    drowsy_queue.append(eyes_closed)
    drowsy_detected = drowsy_queue.count(True) > SMOOTHING_FRAMES // 2

    # ----------------------------
    # Alarm logic
    # ----------------------------
    if drowsy_detected:
        if not alarm_channel.get_busy():
            alarm_channel.play(alarm_sound, loops=-1)
    else:
        if alarm_channel.get_busy():
            alarm_channel.stop()

    # ----------------------------
    # Convert frame to Pygame surface
    # ----------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(frame_surface, (0, 0))

    # ----------------------------
    # Overlay text based on status
    # ----------------------------
    if drowsy_detected:
        status_text = "DROWSINESS DETECTED!"
        color = (255, 0, 0)  # Red
    else:
        status_text = "ACTIVE"
        color = (0, 255, 0)  # Green

    text_surface = font.render(status_text, True, color)
    text_rect = text_surface.get_rect(center=(WEBCAM_WIDTH // 2, 50))
    screen.blit(text_surface, text_rect)

    pygame.display.update()
    pygame.time.delay(10)

    # ----------------------------
    # Handle exit
    # ----------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        running = False

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
pygame.quit()
print("Program exited successfully!")
