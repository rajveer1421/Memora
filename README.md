# ARKA - Alzheimer's Recognition & Knowledge Assistant

## 📖 Project Overview
ARKA is a specialized memory care platform designed to assist Alzheimer's patients in their daily lives while providing peace of mind to their caretakers. The application leverages advanced AI to help patients recognize loved ones and everyday objects, bridging the gap caused by memory loss.

## 🎯 Core Problem & Solution
**The Challenge**: Alzheimer's patients often struggle to recognize family members or recall the function of common objects, leading to anxiety and loss of independence.
**The ARKA Solution**: An intelligent companion app that acts as an "external memory," instantly identifying people and objects via camera and providing personalized audio context (e.g., "This is your daughter, Sarah").

## ✨ Key Features

### For Patients (Simplified Interface)
- **👥 Instant Face Recognition**: Real-time identification of family and friends using simplified camera interface.
- **🔍 Object Recognition**: Identifies common objects (e.g., "keys", "medicine") to help with daily tasks.
- **🔊 Voice Memories**: Plays recorded stories or descriptions in a familiar voice when a person is recognized.
- **📍 SOS & Geofencing**: One-touch emergency alert and automated alerts if the patient wanders outside safe zones.

### For Caretakers (Admin Dashboard)
- **🛡️ Secure Management**: Manage patient profiles and medical data.
- **📸 Face Gallery**: Upload photos to train the AI on specific family members.
- **🎙️ Voice Bank**: Record custom audio messages to be played upon recognition.
- **📊 Activity Logs**: Track recognition events and location history.

## 🏗️ Technical Architecture

ARKA uses a modern, microservices-based architecture to ensure scalability and performance.

### 1. Frontend Layer
- **Framework**: React 18 + Vite
- **Styling**: Vanilla CSS / Tailwind (Responsive & Accessible)
- **Key Libraries**: `react-router-dom`, `lucide-react`
- **Focus**: High-contrast, large-button UI specific for elderly/impaired users.

### 2. Backend & Database
- **Primary Backend**: Node.js + Express
- **Database**: Supabase (PostgreSQL)
  - Uses **pgvector** for storing face embeddings.
- **Authentication**: Supabase Auth (JWT based with Role-Based Access Control).
- **Storage**: Supabase Storage for images and audio files.

### 3. AI/ML Service (The "Brain")
- **Framework**: Python + FastAPI
- **Face Recognition**: **DeepFace** (utilizing ArcFace model) for state-of-the-art facial recognition.
- **Object Detection**: **YOLOv8** (Ultralytics) & MediaPipe for real-time object identification.
- **Logic**:
  1. Receives image from frontend.
  2. Generates vector embeddings.
  3. Performs cosine similarity search against stored embeddings in Supabase.
  4. Returns match with confidence score and associated voice memory.

## 🚀 How It Works (Flow)

1. **Registration**: Caretaker uploads photos of "John (Son)".
2. **Training**: ML Service generates a vector embedding for John's face and saves it to the DB.
3. **Recognition**:
   - Patient points camera at John.
   - App sends frame to ML Service.
   - ML Service calculates vector and finds the closest match in DB.
   - If match > 68% confidence, App displays "John" and plays audio: "This is your son, John."

## 🛠️ Tech Stack Summary
| Component | Technology |
|-----------|------------|
| **Frontend** | React, Vite |
| **API Server** | Python (FastAPI), Node.js (Express) |
| **AI Models** | DeepFace (Face), YOLOv8 (Objects) |
| **Database** | PostgreSQL (Supabase) + pgvector |
| **Deployment** | (Local/Cloud Ready) |


