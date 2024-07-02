# Import các thư viện
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)
# Khởi tạo thư viện Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
lm_list=[]

# Số frame cần lấy cho từng động tác
lable ="Handswing"
lable ="Body"
no_of_frame = 600
# Định nghĩa hàm 
def make_landmark_timestep(results): # Results chứa tất cả tọa độ khung xương
    print (results.pose_landmarks.landmark)
    # Gán tọa độ
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Định nghĩa hàm vẽ điểm nút và đường nối giữa các điểm nút
def draw_landmarks_on_image(mpDraw,results,image):

    # Vẽ các đường nối
    mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    #Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h,w,c =image.shape
        cx = int(lm.x*w)
        cy = int(lm.y*h)
        cv2.circle(image, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    return image
    
# Kiểm tra kết bối đến camera
while len(lm_list)<= no_of_frame:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương và đưa vaò danh sách lm_list
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # Vẽ lên khung xương lên ảnh
            frame = draw_landmarks_on_image(mpDraw, results, frame)

            cv2.imshow("image",frame )
        if cv2.waitKey(1) == ord('q'):
            break

# Ghi vào File VSC
df = pd.DataFrame(lm_list)
df.to_csv(lable +'.txt')
cap.release()
cv2.destroyAllWindows()