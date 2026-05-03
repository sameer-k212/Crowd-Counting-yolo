# 🚀 AI Crowd Analysis System v2.0

Real-time crowd and vehicle detection system using **YOLOv8** + **DeepFace**, with a **FastAPI** Python backend and **Express.js** Node.js frontend.

## ✨ Features

- 👥 **Crowd Detection** — Count people in images, videos, or live webcam
- 🚗 **Vehicle Detection** — Cars, motorcycles, buses, trucks
- 👨👩 **Gender Classification** — Per-person gender detection via DeepFace
- 🔥 **Heatmap Generation** — Gaussian density heatmap overlay
- ⚠️ **Configurable Alerts** — Threshold-based crowd alerts
- 📈 **Video Timeline Chart** — Frame-by-frame crowd count with Chart.js (click to inspect frames)
- 🗄️ **SQLite Persistence** — All analyses saved to database
- 📊 **History & Stats Panel** — View past analyses with summary statistics
- 📥 **CSV Export** — Download full analysis history
- 📝 **Rotating Logs** — File-based logging with auto-rotation

## 🏗️ Architecture

```
Crowd-Counting/
├── main.py           # FastAPI backend — detection, DB, logging, CSV export
├── config.py         # Centralized config from .env
├── database.py       # SQLite handler (analysis, video_frames, settings_store)
├── logger.py         # Rotating file logger
├── server.js         # Express.js frontend server
├── views/
│   └── index.ejs     # Responsive UI (Tailwind CSS + Chart.js)
├── Dockerfile        # Python API container
├── docker-compose.yml
├── requirements.txt
├── package.json
└── .env
```

## 🚀 Quick Start

### Option A — Local (without Docker)

**Prerequisites:** Python 3.8+, Node.js 14+

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Install Node deps
npm install

# 3. Start (runs both servers)
npm start
```

Open http://localhost:3000

### Option B — Docker

```bash
docker-compose up --build
```

Open http://localhost:3000

## ⚙️ Configuration (.env)

```ini
PORT=3000
PYTHON_API_URL=http://127.0.0.1:8001
API_HOST=127.0.0.1
API_PORT=8001
DATABASE_PATH=../crowd_counting.db
LOG_FILE=../logs/app.log
LOG_LEVEL=INFO
ALERT_THRESHOLD=10
MODEL_PATH=yolov8n.pt
```

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload?threshold=10` | Analyze image |
| POST | `/upload-video?threshold=10` | Analyze video |
| POST | `/webcam-frame?threshold=10` | Analyze webcam frame |
| GET | `/history?limit=20` | Get analysis history |
| DELETE | `/history` | Clear history |
| GET | `/stats` | Summary statistics |
| GET | `/export/csv?limit=100` | Download CSV report |

## 🗄️ Database Schema

- **analysis** — stores every detection result (type, counts, confidence, alert, timestamp)
- **video_frames** — per-frame data for video analyses
- **settings_store** — key-value config storage

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI / Detection | YOLOv8 (Ultralytics), DeepFace |
| Backend API | FastAPI, Python 3.11 |
| Frontend Server | Express.js, Node.js |
| UI | EJS, Tailwind CSS, Chart.js |
| Database | SQLite |
| Containerization | Docker, docker-compose |

## 🔐 Security Notes

- No authentication (add JWT before public deployment)
- File uploads validated by content-type
- Temp files use `tempfile.mkstemp()` — no race conditions

## 📦 Production Deployment

```bash
# Build and run with Docker
docker-compose up --build -d

# View logs
docker-compose logs -f
```

For full production: add Nginx reverse proxy, HTTPS, and JWT authentication.

---

**Stack:** FastAPI + Express.js + YOLOv8 + DeepFace + SQLite + Docker
