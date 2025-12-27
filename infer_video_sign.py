from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "weights/sign.pt"
VIDEO_PATH = "input_video.mp4"   # ← 여기에 테스트 영상
CONF = 0.3

PROJECT_DIR = "runs/detect"
RUN_NAME = "video_sign"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=VIDEO_PATH,
    conf=CONF,
    save=True,
    save_txt=False,
    show=False,
    project=PROJECT_DIR,
    name=RUN_NAME
)

print("✅ Video inference finished")
print("📁 Saved to:", Path(PROJECT_DIR) / RUN_NAME)