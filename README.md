# 🧠 Memora — AI Memory Care Platform for Alzheimer's Patients

> *Every face is a memory worth keeping.*

Memora is an AI-powered care platform that helps Alzheimer's patients recognize familiar faces, track emotional well-being, manage daily routines, and stay safe — giving caregivers peace of mind and patients a sense of connection.

---

## 🎯 The Problem

Alzheimer's patients progressively lose the ability to recognize family members, leading to anxiety, fear, and isolation. Caregivers are overwhelmed managing medications, appointments, and safety — often with no technological support tailored to their needs.

## 💡 The Solution

Memora acts as an **external memory system** — a companion app that:
- **Instantly identifies** family members via camera and plays personalized voice messages
- **Detects emotional distress** in speech and triggers reassurance protocols
- **Manages daily routines** with smart medication and meal reminders
- **Tracks mood patterns** to help caregivers intervene proactively

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 👤 **Face Registration** | Upload photos + record voice messages for each family member |
| 🔍 **Live Recognition** | Real-time face identification with confidence scoring and auto voice playback |
| 💬 **Mood Tracking** | NLP-based emotion detection from speech with caregiver alerts |
| 🔔 **Smart Reminders** | Medication, meal, and appointment scheduling with day-of-week targeting |
| 🆘 **SOS & Geofencing** | Safety features roadmap (panic button, safe zone mapping) |
| 📊 **Dashboard** | Centralized view of profiles, reminders, mood trends, and alerts |

---

## 🧬 Technical Architecture

```
┌─────────────────────────────────────────────┐
│                  Frontend                    │
│    Jinja2 Templates + Vanilla JS + CSS       │
│  Camera API · MediaRecorder · SpeechSynth    │
├─────────────────────────────────────────────┤
│               Flask Backend                  │
│         Routes · JSON File Storage           │
├──────────────┬──────────────────────────────┤
│  DeepFace    │   Emotion Classifier          │
│  + ArcFace   │   (Keyword NLP Engine)        │
│  Embeddings  │   Caregiver Alert System      │
└──────────────┴──────────────────────────────┘
```

### Why Embeddings Instead of Raw Images?

**Privacy-first design.** Instead of training a neural network on a patient's photos or storing raw facial images in a database, Memora uses **DeepFace with the ArcFace model** to generate compact vector embeddings (512-dimensional) from each face. Only these numerical vectors are stored — the original photos are not retained for recognition.

**Recognition works via cosine similarity search**: when a new face is captured, its embedding is compared against all stored embeddings. If the cosine distance falls below our 0.68 threshold, we have a match. This approach is:
- **Privacy-preserving** — no raw biometric images stored
- **Fast** — vector comparison is O(n) with minimal compute
- **Accurate** — ArcFace achieves 99.83% accuracy on LFW benchmark

### Multi-Strategy Detection Pipeline

To maximize reliability across different lighting conditions, camera angles, and image quality, Memora uses a **cascading detector strategy**:

| Priority | Detector | Mode | Purpose |
|----------|----------|------|---------|
| 1st | **MediaPipe** | Strict | Highest accuracy face landmark detection |
| 2nd | **OpenCV** | Strict | Fast Haar cascade fallback |
| 3rd | **OpenCV** | Loose | Relaxed detection for difficult angles |
| 4th | **Skip** | — | Direct embedding (assumes face is present) |

Each strategy is tried in order. If one fails, the next is attempted — reducing latency while maintaining high detection rates. This pipeline ensures recognition works even with:
- Low-resolution webcams
- Poor lighting conditions
- Partial face visibility
- Elderly patients who may not face the camera directly

### Mood Tracking Engine

The emotion classifier uses a **keyword-based NLP engine** that maps speech patterns to 7 emotional states: joy, sadness, fear, anger, surprise, disgust, and neutral.

**How it works:**
1. Patient speech is captured (via microphone using Web Speech API or manual text entry)
2. Text is tokenized and matched against curated emotion keyword dictionaries
3. Weighted scoring determines the dominant emotion and confidence level
4. If **fear or sadness** exceeds 70% confidence → caregiver alert is triggered
5. Browser's **SpeechSynthesis API** plays calming reassurance messages

**Future improvement:** The keyword engine is designed as a lightweight placeholder. The architecture supports drop-in replacement with a fine-tuned **DistilRoBERTa** model (`j-hartmann/emotion-english-distilroberta-base`) trained on Alzheimer's patient speech data for higher accuracy.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, Flask |
| **Frontend** | Jinja2, Vanilla JavaScript, CSS |
| **Face Recognition** | DeepFace (ArcFace model) |
| **Face Detection** | MediaPipe, OpenCV (multi-strategy) |
| **Emotion Analysis** | Custom NLP keyword engine |
| **Voice Playback** | Web Speech API (SpeechSynthesis) |
| **Camera/Audio** | getUserMedia, MediaRecorder API |
| **Data Storage** | JSON file-based (embeddings, reminders, mood logs) |
| **Deployment** | Render (Gunicorn) |
| **Math/Vectors** | NumPy, SciPy (cosine similarity) |

---

## 📁 Project Structure

```
Memora/
├── app.py                  ← Flask application (all routes + logic)
├── requirements.txt        ← Python dependencies
├── Procfile                ← Gunicorn start command
├── render.yaml             ← Render deployment blueprint
├── data/
│   ├── embeddings.json     ← Face embedding vectors
│   ├── reminders.json      ← Reminder schedules
│   └── mood_log.json       ← Emotion analysis history
├── static/
│   └── uploads/            ← Uploaded photos and voice recordings
└── templates/
    ├── base.html           ← Base layout + homepage
    ├── register.html       ← Face registration with camera + voice
    ├── recognize.html      ← Live recognition with confidence display
    ├── reminders.html      ← Reminder management (CRUD)
    ├── mood.html           ← Mood analysis + speech-to-text
    ├── sos.html            ← SOS roadmap (in development)
    └── dashboard.html      ← Analytics and overview
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Local Setup

```bash
git clone https://github.com/your-username/Memora.git
cd Memora
pip install -r requirements.txt
python app.py
```

The app will start at `http://localhost:8000`

### Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repository
4. Render auto-detects the `render.yaml` blueprint
5. Deploy 🚀

---

## 📱 How It Works

### 1. Register a Family Member
- Enter their name
- Take a photo (camera or upload)
- Record a personal voice message like *"This is your daughter, Sarah"*
- DeepFace generates a 512-dim embedding and stores it

### 2. Recognize a Visitor
- Point the camera at the visitor
- The system captures a frame and generates an embedding
- Cosine similarity search finds the closest match
- If confidence > 68%: displays the name and plays the voice message
- SpeechSynthesis adds reassurance: *"This is Sarah. You know her well."*

### 3. Track Emotional Health
- Enter or speak what the patient said
- NLP engine classifies the emotion
- If distress is detected → caregiver alert + calming voice message
- All entries are logged for trend analysis

### 4. Manage Daily Reminders
- Set medication, meal, and appointment reminders
- Choose specific days of the week
- All reminders stored and displayed with visual indicators

---

## 🔮 Roadmap

- [ ] Fine-tune DistilRoBERTa on Alzheimer's patient speech data
- [ ] Add SOS panic button with Twilio SMS alerts
- [ ] Implement geofencing with Google Maps API
- [ ] Add multi-patient support with caregiver login
- [ ] Real-time WebSocket notifications
- [ ] PWA support for offline access

---
