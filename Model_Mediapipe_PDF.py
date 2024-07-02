# Import các thư viện
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện và modul của Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hand = mpHands.Hands()
mpBodys = mp.solutions.bodys
body = mpBodys.Bodys()
mpDraw = mp.solutions.drawing_utils

lm_list=[]

# Số frame cho từng lable
lables= ["Handswing","Body"]  # Danh sách lable
for i, lable in enumerate(lables):
    no_of_frame = 300 # số frame cần lấy

# Định nghĩa hàm 
def make_landmark_timestep(results):  # Results chứa tất cả tọa độ khung xương
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Định nghĩa hàm vẽ điểm nút và đường nối giữa các điểm nút
def draw_landmarks_on_image(mpDraw, results, image):
    
    # Vẽ các đường nối
    mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = image.shape
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    return image

# Mở tệp PDF để lưu các khung hình khung xương
pdf_path = "output.pdf"  # Đường dẫn để lưu tệp PDF
pdf_pages = PdfPages(pdf_path)

# Kiểm tra kết nối đến camera
frame_count = 0
while frame_count <= no_of_frame:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương và đưa vào danh sách lm_list
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # Vẽ khung xương lên ảnh
            frame = draw_landmarks_on_image(mpDraw, results, frame)

            # Lưu khung hình khung xương vào tệp PDF
            plt.figure(figsize=(10, 7))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            pdf_pages.savefig()
            plt.close()

            # Hiển thị các frame hình ảnh
            cv2.imshow("image", frame)
        frame_count += 1
        if cv2.waitKey(1) == ord('q'):
            break

# Đóng tệp PDF vừa ghi
pdf_pages.close()

# Ghi vào File CSV
df = pd.DataFrame(lm_list)
df.to_csv(f'{lable}.txt', index=False)
cap.release()
cv2.destroyAllWindows()