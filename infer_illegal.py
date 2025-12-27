from ultralytics import YOLO
import cv2
from pathlib import Path

# =====================
# 설정
# =====================
VIDEO_PATH = "/Users/sunghyunkim/Develop/cv_final/input/input1.mp4"     # 테스트 영상
SIGN_MODEL_PATH = "weights/sign.pt"
CONF_VEHICLE = 0.4
CONF_SIGN = 0.3

OUTPUT_DIR = "runs/detect/illegal_result"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================
# IoU 계산 함수
# =====================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# =====================
# 모델 로드
# =====================
vehicle_model = YOLO("yolov8n.pt")   # pretrained
sign_model = YOLO(SIGN_MODEL_PATH)   # 네가 학습한 모델

# =====================
# 비디오 입출력
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    f"{OUTPUT_DIR}/illegal_result.mp4",
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

# =====================
# 프레임 단위 추론
# =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 차량 검출
    vehicle_results = vehicle_model.predict(frame, conf=CONF_VEHICLE, verbose=False)
    sign_results = sign_model.predict(frame, conf=CONF_SIGN, verbose=False)

    vehicle_boxes = []
    sign_boxes = []

    # 차량 bbox
    for r in vehicle_results:
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck (COCO)
                vehicle_boxes.append(b.xyxy[0].tolist())

    # 장애인 표식 bbox
    for r in sign_results:
        for b in r.boxes:
            sign_boxes.append(b.xyxy[0].tolist())

    # 차량별 판단
    for vbox in vehicle_boxes:
        has_sign = False
        for sbox in sign_boxes:
            if iou(vbox, sbox) > 0.2:
                has_sign = True
                break

        x1, y1, x2, y2 = map(int, vbox)

        if has_sign:
            color = (0, 255, 0)
            label = "LEGAL"
        else:
            color = (0, 0, 255)
            label = "ILLEGAL"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    out.write(frame)

cap.release()
out.release()

print("✅ Illegal parking inference finished")
print("📁 Saved to:", OUTPUT_DIR)