# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "app_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")

# Create folders
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# Model settings
IMG_SIZE = (224, 224)
CLASS_LABELS = ['No Tumor Detected', 'Tumor Detected']
CONF_THRESHOLD = 0.6
