import os
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8001))

DEFAULT_ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", 10))

DATABASE_PATH = os.getenv("DATABASE_PATH", "../crowd_counting.db")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "../logs/app.log")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 50_000_000))
MAX_VIDEO_SIZE = int(os.getenv("MAX_VIDEO_SIZE", 500_000_000))
