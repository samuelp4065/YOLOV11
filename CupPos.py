import cv2
from ultralytics import YOLO

# 훈련된 모델 로드 (훈련된 가중치 파일 경로)
model = YOLO('/home/seongwoo/res/train/YOLOV11_CupSticker/TrainedModel/Detector/weights/best.pt')

# 웹캠 열기
cap = cv2.VideoCapture(2)  # 0은 기본 웹캠

# 3개의 고정된 박스 좌표 (x1, y1, x2, y2)
boxes = [
    (510, 0, 630, 80),  # 첫 번째 박스 (원래 세 번째 박스)
    (370, 0, 490, 80),  # 두 번째 박스 (원래 두 번째 박스)
    (230, 0, 340, 80)   # 세 번째 박스 (원래 첫 번째 박스)
]

while True:
    # 웹캠에서 이미지 캡처
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break
    
    # YOLO 모델을 사용해 실시간 추론
    results = model(frame)  # frame은 현재 웹캠에서 캡처한 이미지
    
    # 고정된 박스들을 이미지에 그리기
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"Box {i+1}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    detected_in_boxes = [False, False, False]  # 각 박스에 컵이 감지되었는지 여부를 추적

    # confidence 0.80 이상인 경우에만 바운딩 박스를 그리기
    if results[0].boxes:
        for box in results[0].boxes:
            if box.conf[0] >= 0.80:
                # 바운딩 박스의 좌표 가져오기 (x1, y1, x2, y2)
                box_x1, box_y1, box_x2, box_y2 = box.xyxy[0]
                
                # 바운딩 박스를 이미지에 그리기
                cv2.rectangle(frame, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{box.conf[0]:.2f}", (int(box_x1), int(box_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 각 고정된 박스 안에 객체가 있는지 확인
                for i, (fixed_x1, fixed_y1, fixed_x2, fixed_y2) in enumerate(boxes):
                    if (fixed_x1 < box_x1 < fixed_x2 and fixed_y1 < box_y1 < fixed_y2) or \
                       (fixed_x1 < box_x2 < fixed_x2 and fixed_y1 < box_y1 < fixed_y2) or \
                       (fixed_x1 < box_x1 < fixed_x2 and fixed_y1 < box_y2 < fixed_y2) or \
                       (fixed_x1 < box_x2 < fixed_x2 and fixed_y1 < box_y2 < fixed_y2):
                        detected_in_boxes[i] = True
                        print(f"컵이 박스 {i+1}에 인식되었습니다 (confidence: {box.conf[0]:.2f}).")

    # 결과를 화면에 표시
    cv2.imshow("Real-time Inference", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()