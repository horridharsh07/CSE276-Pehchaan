# CSE276-Pehchaan — SIH 2026 PS26188
## AI-Based Fake Identity & Document Screening System for Sashastra Seema Bal

> **Smart India Hackathon 2026 · Problem Statement PS26188**  
> Ministry of Home Affairs · Sashastra Seema Bal (SSB) Border Checkposts  
> Indo-Nepal and Indo-Bhutan borders

---

## 🎯 Problem Statement

Sashastra Seema Bal checkpoints process thousands of travellers daily using manual document verification — a slow, error-prone process vulnerable to fake/forged identity documents. The SSB needed an AI-powered, **offline-first** screening system that:

- Detects forged/tampered documents (ELA, copy-move, metadata forensics)
- Verifies the face in the document against the live presenter (deep CNN)
- Reads and validates MRZ, Aadhaar, PAN, DL, passport data via OCR
- Works **100% offline** at remote checkposts with intermittent internet
- Syncs securely to the cloud when connectivity is restored (store-and-forward)
- Maintains a cryptographic hash chain (tamper-evident audit log)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Border Checkpost Device                  │
│                                                              │
│  ┌─────────────────────────┐   ┌──────────────────────────┐ │
│  │  Pehchaan Gateway (Node)│   │  ML Engine (Python/FastAPI│ │
│  │  http://localhost:3000  │◄──►  https://localhost:8001  │ │
│  │  - Auth (JWT + CSRF)    │   │  - Face: SFace CNN (ONNX)│ │
│  │  - UI (React/Next.js)   │   │  - OCR: Tesseract        │ │
│  │  - Offline outbox       │   │  - Tamper: ELA + ORB     │ │
│  │  - Hash chain           │   │  - mTLS secured          │ │
│  └─────────────────────────┘   └──────────────────────────┘ │
│                │                                             │
└────────────────┼─────────────────────────────────────────────┘
                 │  (sync when internet available)
                 ▼
         ┌───────────────┐
         │ Neon PostgreSQL│
         │ (cloud, port  │
         │  443/WebSocket│
         └───────────────┘
```

### Key Design Choices

| Feature | Implementation | Why |
|---|---|---|
| **Offline-first** | Local SQLite outbox → cloud sync | SSB checkposts lose internet frequently |
| **Face verification** | OpenCV SFace CNN + YuNet detector | 36.9 MB ONNX, runs on CPU, no GPU needed |
| **Deep scan preprocessing** | CLAHE + denoising + sharpening + white balance | Document photos are scanned/printed, different quality from selfies |
| **Port 443 DB** | `@neondatabase/serverless` WebSocket | College/ISP firewalls block port 5432 |
| **mTLS** | Mutual TLS between Gateway ↔ ML Engine | Prevents rogue clients from calling the ML API directly |
| **Hash chain** | SHA-256 chained scan records | Tamper-evident audit trail per ICAO/MHA guidelines |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **Tesseract OCR** (`choco install tesseract` on Windows)

### 1. ML Engine

```powershell
cd Pehchaan/backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# Download face models (required for deep-scan face verification)
python scripts/download_models.py

# Start engine
.\.venv\Scripts\python run_engine.py --port 8001
```

### 2. Gateway + UI

```powershell
cd Pehchaan/Pehchaan
npm install
npm run dev
# Open http://localhost:3000
```

### 3. One-click launcher
```powershell
.\start_servers.bat
```

---

## 🔐 Demo Credentials

| Role | Email | Password |
|---|---|---|
| Officer | `officer@pehchaan.gov.in` | `officer123` |
| Post Incharge | `incharge@pehchaan.gov.in` | `incharge123` |
| Admin | `admin@pehchaan.gov.in` | `ABC@12345678` |

---

## 🧠 Deep-Scan Face Verification

The face pipeline runs three backends in order (strongest first):

1. **OpenCV SFace** (`face_recognition_sface_2021dec.onnx`) — 128-D cosine embedding  
   - Cross-modal calibrated: match ≥ 0.42, mismatch < 0.26  
   - Multi-scale embedding (112px + 160px averaged)  
   - 70/30 ensemble blend with dlib when available  

2. **dlib 128-D** (if `face_recognition` package installed)

3. **Enhanced CLAHE + multi-scale ORB + NCC fallback** (always available)

**Document preprocessing pipeline** (applied before embedding):
```
Upscale (if < 160px) → White balance → NLM Denoising → Unsharp mask → CLAHE
```

---

## 📊 Risk Scoring

| Signal | Weight | Description |
|---|---|---|
| Document validation | 30% | MRZ, format rules, date logic |
| Tampering forensics | 35% | ELA + copy-move + metadata |
| Face match | 20% | Document photo vs live selfie |
| Records cross-check | 15% | Prior screenings, issued DB |

**Verdict bands**: 0–30 → CLEAR · 31–65 → REVIEW · 66–100 → REJECT

All thresholds are configurable via environment variables (see `.env.example`).

---

## 📁 Project Structure

```
CSE276PEHCHAANproject/
├── Pehchaan/
│   ├── backend/          # ML Engine (Python/FastAPI)
│   │   ├── app/
│   │   │   ├── modules/
│   │   │   │   ├── face/         # Deep-scan face detection + verification
│   │   │   │   ├── ocr/          # Tesseract OCR + MRZ parser
│   │   │   │   ├── tampering/    # ELA + copy-move + metadata
│   │   │   │   ├── risk/         # Risk scoring engine
│   │   │   │   ├── validation/   # Document rules engine
│   │   │   │   └── records/      # Cross-check / forgery registry
│   │   │   └── api/              # FastAPI routes
│   │   ├── models/face/          # ONNX model weights (download separately)
│   │   └── run_engine.py
│   └── Pehchaan/         # Gateway + React UI (Next.js/Node.js)
│       ├── server/        # Express server, auth, Neon DB, outbox
│       └── src/           # React components
├── synopsis_figures/      # Academic figures (300 DPI)
├── DEMO_TEST_DOCUMENTS/   # Sample test documents
└── start_servers.bat      # One-click launcher
```

---

## 🌐 Deployment

See `deploy/` for Docker + Render configuration.

```bash
docker-compose up      # local test
# Push to Render via render.yaml
```

---

## 📚 Academic

- **Synopsis**: `PEHCHAAN_SIH26188_Project_Synopsis.docx`
- **Figures**: `synopsis_figures/` (7 publication-quality figures)
- **Team**: SIH 2026 — CSE276
