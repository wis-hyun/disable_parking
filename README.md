## 장애인 주차구역 불법 주정차 감지 시스템 (disable_parking)

본 프로젝트는 YOLOv8 기반 객체 탐지 모델을 활용하여
장애인 주차구역 내 불법 주정차 차량을 자동으로 감지하는 AI 시스템입니다.

차량(Car)과 표지판(Sign)을 동시에 인식하여
단순 차량 탐지를 넘어 장애인 전용 주차구역 여부까지 판단하는 것을 목표로 합니다.

⸻

### 프로젝트 개요
		
•	문제 정의
장애인 주차구역의 불법 점유는 사회적 문제이지만,
인력 기반 단속에는 한계가 존재합니다.


•	해결 방법
객체 탐지 기반 AI 모델을 활용해
▶ 차량 존재 여부
▶ 장애인 주차 표지판 여부
를 종합적으로 분석하여 불법 주차를 판단합니다.

⸻

### 사용 기술
	•	Model: YOLOv8 (Ultralytics)
	•	Language: Python
	•	Framework / Library:
	•	PyTorch
	•	OpenCV
	•	Ultralytics YOLO
	•	Input: 이미지 / 영상
	•	Output: 불법 주정차 여부 판별 결과

⸻

프로젝트 구조
```
disable_parking/
│
├── frontend/              # 프론트엔드 (대시보드 / UI)
│
├── weights/               # 학습된 모델 가중치
│
├── fix_labels.py          # 라벨 전처리 및 수정 스크립트
├── train_sign.py          # 표지판(Sign) 모델 학습 코드
│
├── infer_sign.py          # 표지판 단일 이미지 추론
├── infer_illegal.py       # 불법 주차 여부 판단 로직
├── infer_combined.py      # 차량 + 표지판 종합 추론
├── infer_video_sign.py    # 영상 기반 표지판 추론
│
├── train.log              # 학습 로그
├── yolov8n.pt             # YOLOv8 기본 pretrained weight
│
└── README.md
```

⸻

실행 방법
```
1️⃣ 환경 설정

pip install ultralytics opencv-python torch

2️⃣ 모델 학습 (표지판)

python train_sign.py

3️⃣ 이미지 추론

python infer_combined.py --img path/to/image.jpg

4️⃣ 영상 추론

python infer_video_sign.py --video path/to/video.mp4
```

⸻

### 불법 주차 판단 로직 
	1.	차량 객체 탐지
	2.	장애인 주차 표지판 탐지
	3.	차량 + 장애인 표지판 동시 검출 시
	→ 장애인 주차구역 점유로 판단
	4.	장애인 차량 여부 미확인 시 불법 주정차로 분류

⸻

### 기대 효과
	•	장애인 주차구역 불법 점유 감소
	•	인력 단속 부담 완화
	•	스마트 시티 · 스마트 파킹 시스템 확장 가능

⸻

개발자
	•	wis-hyun
	•	AI / Computer Vision Project

⸻

참고
	•	YOLOv8: https://github.com/ultralytics/ultralytics
