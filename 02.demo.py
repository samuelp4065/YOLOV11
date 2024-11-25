# from ultralytics import YOLO

# # 위에서 Trained된 모델 불러오기
# model = YOLO('yolov8n.pt')

# # inference 하고 싶은 이미지 불러오기
# source = 'path/to/image.jpg'

# # Run inference on the source
# results = model(source)  # list of Results objects

import cv2
from ultralytics import YOLO

# 훈련된 모델 로드 (훈련된 가중치 파일 경로)
model = YOLO('/home/seongwoo/res/objectDetection/Results/CupDetection2/weights/best.pt')

# 웹캠 열기
cap = cv2.VideoCapture(2)  # 0은 기본 웹캠

while True:
    # 웹캠에서 이미지 캡처
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 열 수 없습니다.")
        break
    
    # YOLO 모델을 사용해 실시간 추론
    results = model(frame)  # frame은 현재 웹캠에서 캡처한 이미지
    
    # 추론 결과에서 바운딩 박스를 그리기
    # results.plot()은 바운딩 박스를 그린 이미지 반환
    annotated_frame = results[0].plot()  # 첫 번째 결과에서 바운딩 박스를 그려서 가져옵니다.

    if results[0].boxes:
        print("스티커가 있습니다")
        # cv2.putText(annotated_frame, "스티커", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
        print(results[0].boxes)
    # 결과를 화면에 표시
    cv2.imshow("Real-time Inference", annotated_frame)

    
    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
