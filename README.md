                           Drowsiness-Detection-System
description: >
  Real-time driver drowsiness detection using YOLO and Pygame. 
  Detects eye closure and triggers an alarm if drowsiness is detected. 
  Features smoothing and confidence-based detection to reduce false alarms.
tags:
  - Python
  - OpenCV
  - YOLO
  - Pygame
  - DrowsinessDetection
  - ComputerVision
  - RealTime
  - AI
features:
  - Real-time detection of open and closed eyes using YOLO
  - Confidence-based detection to reduce false alarms
  - Smoothing over multiple frames for reliable detection
  - Visual feedback with status: "ACTIVE" when alert, "DROWSINESS DETECTED!" when sleepy
  - Alarm triggered automatically on prolonged eye closure
  - Works with standard webcam; no special hardware required
social_impact: >
  Drowsy driving is a leading cause of road accidents worldwide. This system improves road safety by:
  - Reducing accidents caused by driver fatigue
  - Protecting lives by alerting drivers before fatigue leads to dangerous situations
  - Promoting awareness about the importance of alertness during driving
installation:
  steps:
    - Clone the repository:
        command: git clone https://github.com/NandiniMittal1311/Drowsiness-Detection-System.git
    - Navigate into the project directory:
        command: cd Drowsiness-Detection-System
    - Install required packages:
        command: pip install ultralytics opencv-python pygame requests numpy
    - Add alarm file:
        instruction: Place 'alarm.wav' in the project directory
usage:
  steps:
    - Run the main script:
        command: python drowsiness_detection.py
    - Webcam window behavior:
        - "ACTIVE" in green when eyes are open
        - "DROWSINESS DETECTED!" in red when eyes are closed
    - Alarm will play automatically when drowsiness is detected
    - Press 'q' or close the window to exit
repository_structure:
  - drowsiness_detection.py: Main script
  - alarm.wav: Alarm sound file
  - open_closed_eye_model.pt: YOLO eye detection model
  - README.md: Project documentation
  - requirements.txt: Optional dependencies
how_it_works:
  - Captures video frames from webcam
  - YOLO detects eyes as open or closed with confidence scores
  - Computes normalized probability to reduce misclassification of wide-open eyes
  - Maintains frame history (smoothing) to ensure persistent drowsiness triggers alarm
  - Updates display with status text and alarm sound
potential_extensions:
  - Integrate with vehicle dashboards or smart glasses
  - Add logging to record drowsiness events
  - Combine with heart rate or facial landmarks for more robust detection
license: MIT

