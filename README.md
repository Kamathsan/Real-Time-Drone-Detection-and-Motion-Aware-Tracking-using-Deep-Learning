Syracuse: Real Time Drone Detection and Motion Aware Tracking using DeepLearning🎯
📌 Overview

This project implements an AI-powered autonomous turret system capable of real-time drone detection, motion-aware tracking, and servo-based actuation.
The system uses the YOLOv11m object detection model for accurate detection, combined with PID-controlled pan–tilt movement driven by an Arduino-based hardware setup.

The project demonstrates the integration of computer vision, deep learning, control systems, and embedded hardware into a unified real-time system.

🚀 Key Features

🔍 Real-Time Drone Detection using YOLOv11m

🎯 Motion-Aware Tracking with velocity prediction

🎛 PID-Based Servo Control for smooth and stable movement

🤖 Arduino Integration via serial communication

🔄 Search & Reacquisition Mode when the target is lost

⚡ GPU Acceleration (CUDA) support

📊 Live FPS, Bounding Boxes & Tracking Status UI

🧠 System Workflow

Capture live video feed from the camera

Detect drones using YOLOv11m

Track objects using ByteTrack

Predict target motion using velocity estimation

Compute servo corrections using PID controllers

Send commands to Arduino for pan–tilt actuation

Enter scanning mode if target is lost

🖼️ Project Screenshots & Setup Images

📌 Insert your images here

Hardware setup (Turret, Servos, Arduino)

Camera feed with detection & tracking UI

Search mode vs Tracking mode visualization

/assets
 ├── hardware_setup.jpg
 ├── tracking_ui.jpg
 ├── search_mode.jpg

🛠 Tech Stack
Software

Python 3.8+

PyTorch

Ultralytics YOLOv11m

OpenCV

NumPy

ByteTrack

PySerial

Hardware

Arduino (Uno / Nano / Mega)

Pan–Tilt Servo Motors

USB Camera / Webcam

Power Supply

Laser Module / Indicator (optional)

📂 Project Structure
├── best(yolo11m).pt              # Trained YOLOv11m model weights
├── dronedetection(yolo_11m).ipynb
├── main_tracking.py              # Main detection + tracking logic
├── arduino_controller.ino        # Arduino servo control code
├── assets/                       # Images for README
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-autonomous-turret.git
cd ai-autonomous-turret

2️⃣ Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

3️⃣ Install Dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pyserial


⚠️ If using GPU, install the CUDA-compatible PyTorch version.

4️⃣ Connect Hardware

Connect Arduino via USB

Upload the Arduino .ino file using Arduino IDE

Connect servos to PWM pins (as defined in the Arduino code)

Ensure external power for servos if required

5️⃣ Configure Model & Parameters

Edit the configuration section in the Python file:

MODEL_PATH = "path/to/best(yolo11m).pt"
BAUD_RATE = 115200
DETECTION_CONF = 0.5


Adjust servo limits and PID values as needed:

PID_X_KP = 0.9
PID_X_KI = 0.02
PID_X_KD = 0.4

▶️ Running the System
python main_tracking.py

Controls

ESC → Exit program

R → Return turret to home position

🧪 Modes of Operation
🔎 Tracking Mode

Activated when a drone is detected

Laser/indicator turns ON

Motion-aware prediction improves response time

🔄 Search Mode

Activated when the target is lost

Turret follows a spiral scan pattern

Automatically reacquires target when detected

📈 Performance Highlights

Stable real-time tracking with minimal jitter

Smooth servo transitions using PID control

Reliable detection using YOLOv11m medium model

Adaptive tracking for slow and fast-moving targets

🧩 Applications

Drone surveillance systems

Defense & perimeter monitoring

Smart security solutions

Robotics & AI research

Academic final-year projects

🔮 Future Improvements

Multi-object tracking & prioritization

Thermal + RGB fusion

Web-based monitoring dashboard

Edge deployment (Jetson Nano / Xavier)

Autonomous threat classification

👤 Author

Shashank Kamath
Final Year B.E. – Computer Science Engineering

⭐ Acknowledgements

Ultralytics YOLO

OpenCV Community

Arduino Open Source Ecosystem

⭐ If you find this project useful, consider giving it a star!
