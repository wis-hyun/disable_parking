from ultralytics import YOLO
from datetime import datetime
import sys

# ===== 설정 =====
MODEL_NAME = "yolov8n.pt"
DATA_YAML = "disabled_sign_detection/data.yaml"
EPOCHS = 50
IMGSZ = 416
BATCH = 16
RUN_NAME = "sign_yolo"

# ===== 로그 파일 설정 =====
log_file = open("train.log", "w")
sys.stdout = log_file
sys.stderr = log_file

print("===== Disabled Sign Detection Training =====")
print(f"Start time: {datetime.now()}")
print(f"Model: {MODEL_NAME}")
print(f"Data: {DATA_YAML}")
print(f"Epochs: {EPOCHS}, Image Size: {IMGSZ}, Batch: {BATCH}")
print("============================================\n")

# ===== 모델 로드 =====
model = YOLO(MODEL_NAME)

# ===== 학습 =====
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    name=RUN_NAME
)

print("\n============================================")
print(f"End time: {datetime.now()}")
print("Training finished successfully.")

log_file.close()