import cv2
import os

FolderName = "Cup_V2"
os.makedirs(FolderName, exist_ok=True)

# 카메라 초기화
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

image_count = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 프레임을 화면에 표시
    cv2.imshow("Camera", frame)

    # 스페이스바를 누르면 이미지 저장
    if cv2.waitKey(1) & 0xFF == ord(' '):
        image_filename = os.path.join(FolderName, f"Cup_{image_count}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"이미지가 저장되었습니다. : {image_filename}")
        image_count += 1
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()