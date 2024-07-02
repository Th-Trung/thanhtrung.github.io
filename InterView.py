# Import các thư viện
import cv2
import keras
import mediapipe as mp
import numpy as np
import pandas as pd
import threading

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load model 
model = keras.models.load_model('Train_model.h5')

lm_list=[]

# Số frame cần lấy cho từng động tác
lable ="Handswing"
lable ="Body"
no_of_frame = 10

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

# Hàm vẽ chữ lên ảnh
def draw_class_image(lable, image):
    font = cv2.FONT_HERSHEY_COMPLEX
    bottomLeftCornerOfTest = (10,30)
    fontScale = 1
    fontColor = (80,199,199)
    thickness = 2
    lineType = 2
    cv2.putText(image, lable, bottomLeftCornerOfTest,font,fontScale, thickness, lineType)
    return image

# Định nghĩa hàm detect
def detect (model, lm_list):
    global lable
    lm_list = np.array(lm_list)
    lm_list = np.explan_dims(lm_list, axis =0)
    print(lm_list.shape)
    results=model.predict(lm_list)
    print(results)
    if len(lm_list) == 10:
        lm_list_tensor = np.expand_dims(lm_list,axis=0)
        lm_list = []
        action_result = model.predict(lm_list_tensor)
        if action_result[0][0]>0.5:
            lable = "Handswing"
        else:
            lable = "Body"

    return lable
   
# Kiểm tra kết bối đến camera
while len(lm_list)<= no_of_frame:
    ret, frame = cap.read()
    if ret:
    # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=pose.process(frameRGB)
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương và đưa vaò danh sách lm_list
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # Đưa vào model nhận diện
            if len(lm_list)==10:
                thread_1 = threading.Thread(target = detect, args = (model, lm_list))
                thread_1.start()
                lm_list=[]

            # Vẽ lên khung xương lên ảnh
            frame = draw_landmarks_on_image(mpDraw, results, frame)

        frame = draw_class_image(lable, frame)
        cv2.imshow("Image",frame )
        if cv2.waitKey(1) == ord('q'):
            break

# Ghi vào File VSC
df = pd.DataFrame(lm_list)
df.to_csv(lable +'.txt')
cap.release()
cv2.destroyAllWindows()