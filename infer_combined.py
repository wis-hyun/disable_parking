from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime


# ===============================
# 설정
# ===============================
VIDEO_PATH = "/Users/sunghyunkim/25-2-CV-final/cv_final/input/input1.mp4"
SIGN_MODEL_PATH = "weights/sign.pt"
CAR_MODEL_NAME = "yolov8n.pt"

CONF_CAR = 0.4
CONF_SIGN = 0.3

OUTPUT_DIR = "runs/detect/combined_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# ROI 선택 (마우스)
# ===============================
roi_points = []

def mouse_callback(event, x, y, flags, param):
    global roi_points, roi_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        cv2.circle(roi_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            roi_frame, str(len(roi_points)),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )
        cv2.imshow("Select Disabled Parking ROI", roi_frame)

# ===============================
# 비디오 로드 (ROI 먼저)
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

ret, roi_frame = cap.read()
if not ret:
    raise RuntimeError(" Cannot read first frame")

cv2.namedWindow("Select Disabled Parking ROI")
cv2.setMouseCallback("Select Disabled Parking ROI", mouse_callback)

print("▶ 장애인 주차구역 ROI를 시계방향으로 4번 클릭")
print("▶ 다 찍고 Enter / 취소는 ESC")

cv2.imshow("Select Disabled Parking ROI", roi_frame)

while True:
    key = cv2.waitKey(1)
    if key == 13 and len(roi_points) == 4:  # Enter
        break
    elif key == 27:  # ESC
        print(" ROI 선택 취소")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

DISABLED_ROI = np.array(roi_points, dtype=np.int32)
print(f"DISABLED_ROI = {DISABLED_ROI.tolist()}")

# ===============================
# 모델 로드
# ===============================
car_model = YOLO(CAR_MODEL_NAME)
sign_model = YOLO(SIGN_MODEL_PATH)
print(" Models loaded")

# ===============================
# 비디오 속성
# ===============================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1:
    fps = 30.0

input_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

out_filename = f"{input_name}_result_{timestamp}.mp4"
out_path = os.path.join(OUTPUT_DIR, out_filename)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# ===============================
# 프레임 단위 추론
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 시각화
    cv2.polylines(frame, [DISABLED_ROI], True, (0, 255, 255), 2)
    cv2.putText(
        frame, "DISABLED ZONE",
        (DISABLED_ROI[0][0], DISABLED_ROI[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # 차량 검출
    car_results = car_model.predict(
        source=frame,
        conf=CONF_CAR,
        classes=[2, 5, 7],
        verbose=False
    )

    # 장애인 표식 검출
    sign_results = sign_model.predict(
        source=frame,
        conf=CONF_SIGN,
        verbose=False
    )

    car_boxes = car_results[0].boxes
    sign_boxes = sign_results[0].boxes

    # sign bbox 리스트
    sign_list = []
    if sign_boxes is not None:
        for b in sign_boxes:
            sign_list.append(tuple(map(int, b.xyxy[0])))

    # 차량별 판별
    if car_boxes is not None:
        for box in car_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = y2 - 5   # bbox 하단 기준 (바퀴 위치 근사)

            in_roi = cv2.pointPolygonTest(DISABLED_ROI, (cx, cy), False) >= 0

            if not in_roi:
                color = (180, 180, 180)
                status = "OUTSIDE"
            else:
                has_sign = False
                car_height = y2 - y1
                windshield_y_limit = y1 + int(car_height * 0.4)  # 차량 상단 40%

                for sx1, sy1, sx2, sy2 in sign_list:
                    sign_cx = (sx1 + sx2) // 2
                    sign_cy = (sy1 + sy2) // 2

                # sign이 차량 bbox 안에 있고
                # 차량 상단(유리 위치)에 있을 때만 인정
                    if (x1 <= sign_cx <= x2) and (y1 <= sign_cy <= windshield_y_limit):
                        has_sign = True
                        break

                if has_sign:
                    color = (0, 255, 0)
                    status = "LEGAL"
                else:
                    color = (0, 0, 255)
                    status = "ILLEGAL"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(
                frame, status,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

    # sign 시각화
    if sign_boxes is not None:
        for box in sign_boxes:
            sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
            cv2.putText(
                frame, "SIGN",
                (sx1, max(20, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

    # 결과 영상 저장
    writer.write(frame)

    # ===============================
    # 실시간 화면 출력 (시연용)
    # ===============================
    cv2.imshow("Live Illegal Parking Detection", frame)

    # q 또는 ESC 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("⏹ 사용자 종료")
        break

# ===============================
# 종료
# ===============================
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Combined inference finished")
print(f" Saved to: {out_path}")