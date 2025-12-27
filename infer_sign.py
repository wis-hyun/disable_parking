from ultralytics import YOLO
from pathlib import Path

# ===== 설정 =====
MODEL_PATH = "weights/sign.pt"
SOURCE = "disabled_sign_detection/valid/images"
CONF = 0.3

PROJECT_DIR = "runs/detect"   # CV_FINAL 안
RUN_NAME = "predict_sign"

# ===== 모델 로드 =====
model = YOLO(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

# ===== 추론 =====
results = model.predict(
    source=SOURCE,
    conf=CONF,
    save=True,
    save_txt=False,
    show=False,
    project=PROJECT_DIR,
    name=RUN_NAME
)

print("✅ Inference finished")
print("📁 Results saved in:", Path(PROJECT_DIR) / RUN_NAME)