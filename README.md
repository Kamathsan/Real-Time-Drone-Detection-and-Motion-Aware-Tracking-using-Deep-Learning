## Syracuse: Real Time Drone Detection and Motion Aware Tracking using DeepLearning🎯
## Overview
This project implements an AI-powered autonomous turret system capable of real-time drone detection, motion-aware tracking, and servo-based actuation. The system uses the **YOLOv11m object detection model** for accurate detection, combined with PID-controlled pan–tilt movement driven by an Arduino-based hardware setup.

The project demonstrates the integration of computer vision, deep learning, control systems, and embedded hardware into a unified real-time system.

---

## Key Features
- Real-time drone detection using YOLOv11m  
- Motion-aware object tracking with velocity prediction  
- PID-based servo control for smooth and stable movement  
- Arduino integration via serial communication  
- Automatic search and target reacquisition mode  
- GPU acceleration support (CUDA)  
- Live FPS, bounding boxes, and tracking status UI  

---

## System Workflow
1. Capture live video feed from the camera  
2. Detect drones using YOLOv11m  
3. Track objects using ByteTrack  
4. Predict target motion using velocity estimation  
5. Compute servo corrections using PID controllers  
6. Send commands to Arduino for pan–tilt actuation  
7. Enter scanning mode if the target is lost  

---

## Project Screenshots and Setup
images here!



---

## Tech Stack

### Software
- Python 3.8+
- PyTorch
- Ultralytics YOLOv11m
- OpenCV
- NumPy
- ByteTrack
- PySerial

### Hardware
- Arduino (Uno / Nano / Mega)
- Pan–Tilt Servo Motors (SG995 with metal gears preferred)
- USB Camera / Webcam
- External power supply for servos (2x3.7v rechargable Li-ion Batteries recommended) 
- Laser module or indicator (optional)

---

## Project Structure
├── best(yolo11m).pt # Trained YOLOv11m model weights
├── dronedetection(yolo_11m).ipynb
├── main_tracking.py # Main detection and tracking script
├── arduino_controller.ino # Arduino servo control code
├── assets/ # README images
└── README.md


---

## Installation and Setup

### 1. Clone the Repository
git clone https://github.com/Kamathsan/Real-Time-Drone-Detection-and-Motion-Aware-Tracking-using-Deep-Learning
---
### 2. download the best.pt file from the below link
Best.pt(YOLO-11m) file: https://drive.google.com/file/d/1bn83zFnG0gpNwWHVGzj6M5aNa_r3izb3/view?usp=sharing
---
### 3. Create a Virtual Environment (Recommended)
python -m venv venv

Activate the environment:

**Windows**
venv\Scripts\activate

**Linux / macOS**
source venv/bin/activate

---

### 4. Install Dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pyserial


> For GPU support, install the CUDA-compatible version of PyTorch.

---

### 5. Hardware Setup
- Connect the Arduino via USB  
- Upload the Arduino `.ino` file using Arduino IDE  
- Connect the servos to PWM pins as defined in the Arduino code  
- Use an external power supply if required  

---

### 6. Configuration
Update the configuration section in the Python script:
MODEL_PATH = "path/to/best(yolo11m).pt"
BAUD_RATE = 115200
DETECTION_CONF = 0.5

PID tuning parameters can be adjusted based on hardware response:
PID_X_KP = 0.9
PID_X_KI = 0.02
PID_X_KD = 0.4


---

## Running the Project
python main_tracking.py


### Controls
- ESC – Exit the application  
- R – Return turret to home position  

---

## Modes of Operation

### Tracking Mode
- Activated when a drone is detected  
- Target is locked and followed smoothly  
- Laser/indicator is enabled  

### Search Mode
- Activated when the target is lost  
- Turret performs a controlled scanning pattern  
- Automatically switches back to tracking on detection  

---

## Performance Highlights
- Stable real-time tracking with minimal jitter  
- Smooth servo movement using PID control  
- Accurate detection using YOLOv11m medium model  
- Adaptive tracking for slow and fast-moving targets  

---

## Applications
- Drone surveillance and monitoring  
- Defense and perimeter security systems  
- Smart security automation  
- Robotics and AI research  
- Academic final-year engineering projects  

---

## Future Enhancements
- Multi-object tracking and prioritization  
- Thermal and RGB camera fusion  
- Web-based monitoring dashboard  
- Edge deployment (Jetson Nano / Xavier)  
- Autonomous threat classification  

---

## Author
Shashank Kamath  
Computer Science and Design- Canara Engineering College

---

## Acknowledgements
- Ultralytics YOLO  
- OpenCV Community  
- Arduino Open Source Ecosystem  

---

If you find this project useful, consider giving the repository a star⭐  :)

