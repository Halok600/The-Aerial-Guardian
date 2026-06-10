# 🚁 The Aerial Guardian

### AI-Powered Drone Surveillance & Multi-Object Tracking System

> Transforming aerial footage into actionable intelligence using Computer Vision, Deep Learning, and Real-Time Object Tracking.

---

## 📖 Overview

**The Aerial Guardian** is an intelligent drone surveillance system capable of detecting, tracking, and monitoring objects in aerial videos in real time.

The project leverages a custom-trained **YOLOv8 detector** combined with **ByteTrack Multi-Object Tracking** to accurately identify and maintain object identities across video frames captured by drones.

Designed using the **VisDrone dataset**, this system aims to address challenges commonly faced in aerial surveillance such as:

* Small object detection
* Camera motion caused by drone movement
* Occlusions and crowded scenes
* Real-time processing requirements
* Persistent object tracking

The system can be applied in:

* 🚔 Security & Surveillance
* 🚦 Traffic Monitoring
* 🌾 Agricultural Observation
* 🚨 Disaster Response
* 🏙️ Smart City Monitoring
* 🎯 Search & Rescue Operations

---

# ✨ Features

✅ Real-Time Object Detection

✅ Multi-Object Tracking (MOT)

✅ Drone Camera Motion Compensation

✅ Custom YOLOv8 Training

✅ High-Resolution Video Processing

✅ Persistent Object IDs

✅ Bounding Box Visualization

✅ Performance Evaluation Metrics

✅ Optimized for Aerial Footage

---

# 🏗️ System Architecture

```text
Drone Video Input
        │
        ▼
Preprocessing
        │
        ▼
YOLOv8 Object Detection
        │
        ▼
ByteTrack Association
        │
        ▼
Camera Motion Compensation
        │
        ▼
Object ID Assignment
        │
        ▼
Visualization & Output Video
```

---

# 🛠️ Technology Stack

## Programming Language

* Python

## Deep Learning

* YOLOv8
* PyTorch

## Computer Vision

* OpenCV

## Tracking

* ByteTrack

## Dataset

* VisDrone MOT Dataset

## Visualization

* OpenCV Rendering

## Training Environment

* Google Colab
* Jupyter Notebook

---

# 📂 Project Structure

```text
The-Aerial-Guardian/
│
├── data/
│   ├── images/
│   ├── labels/
│
├── models/
│   ├── best.pt
│
├── videos/
│   ├── input/
│   ├── output/
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── track.py
│   ├── evaluate.py
│
├── results/
│   ├── metrics/
│   ├── visualizations/
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Model Details

### Detector

**YOLOv8n-P2**

The YOLOv8n-P2 architecture was selected due to its enhanced capability for detecting small objects in aerial imagery.

Advantages:

* Lightweight architecture
* Fast inference speed
* Better small-object detection
* Suitable for real-time deployment

---

### Tracker

**ByteTrack**

ByteTrack associates detected objects across consecutive frames and maintains consistent object identities.

Benefits:

* Robust tracking performance
* Handles occlusions effectively
* Maintains stable IDs
* Works efficiently with drone footage

---

# 📊 Dataset

### VisDrone Dataset

The project was trained and evaluated using the VisDrone benchmark dataset.

Dataset contains:

* Aerial Images
* Video Sequences
* Object Detection Annotations
* Multi-Object Tracking Labels

Object Categories Include:

* Pedestrian
* Person
* Bicycle
* Car
* Van
* Truck
* Bus
* Motorcycle
* Others

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/The-Aerial-Guardian.git

cd The-Aerial-Guardian
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

### Train the Model

```bash
python train.py
```

### Run Detection

```bash
python detect.py
```

### Run Tracking

```bash
python track.py
```

### Evaluate Performance

```bash
python evaluate.py
```

---

# 📈 Evaluation Metrics

The system can be evaluated using:

* mAP (Mean Average Precision)
* Precision
* Recall
* MOTA
* MOTP
* IDF1 Score
* FPS (Frames Per Second)

---

# 🔬 Challenges Addressed

Drone-based vision systems introduce unique challenges:

### Small Objects

Objects often occupy very few pixels in aerial footage.

### Camera Motion

Continuous drone movement can affect tracking accuracy.

### Occlusions

Objects frequently overlap in crowded scenes.

### Scale Variations

Object sizes change significantly depending on altitude.

The Aerial Guardian addresses these issues through optimized detection and tracking pipelines.

---

# 🎯 Future Improvements

* Live Drone Feed Integration
* Edge Deployment on NVIDIA Jetson
* Object Re-Identification (ReID)
* Crowd Density Estimation
* Suspicious Activity Detection
* Multi-Drone Coordination
* Web Dashboard for Monitoring
* AI-Based Threat Detection

---

# 📸 Sample Output

```text
Frame 124

ID 01 → Person
ID 02 → Person
ID 03 → Car
ID 04 → Bicycle

Tracking Status: Active
```

---

# 👨‍💻 Author

### Priyanshu Tiwari

B.Tech Computer Science Engineering

Passionate about:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Drone Technology
* Software Development

---

# ⭐ Final Note

This project was built as part of my exploration into the intersection of **Computer Vision, Deep Learning, and Autonomous Aerial Systems**. The goal was not only to create a robust object detection and tracking pipeline but also to understand the real-world challenges involved in deploying AI systems on aerial platforms.

If you find this project interesting, consider giving it a ⭐ and feel free to contribute or suggest improvements.
