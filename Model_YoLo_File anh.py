# Import các thư viện
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

# Đường dẫn đến thư mục chứa ảnh
image_folder = 'C:/Users/TRUNG/Desktop/Doan/DataSet/Bao luc/12.jpg'  # Thay thế bằng đường dẫn thư mục của bạn

# Đường dẫn đến tệp cấu hình và trọng số của mô hình YOLO
yolo_config = 'yolov10.cfg'  # Thay thế bằng đường dẫn tệp cấu hình YOLO v10
yolo_weights = 'yolov10.weights'  # Thay thế bằng đường dẫn tệp trọng số YOLO v10
yolo_classes = 'yolov10.names'  # Thay thế bằng đường dẫn tệp chứa tên các lớp đối tượng YOLO v10

# Đọc tên các lớp từ tệp
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Khởi tạo mô hình YOLO
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Định nghĩa hàm vẽ các hộp bao quanh đối tượng nhận diện
def draw_labels_and_boxes(image, boxes, confidences, class_ids, classes):
    for (box, confidence, class_id) in zip(boxes, confidences, class_ids):
        (x, y, w, h) = box
        color = (0, 255, 0)
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Mở tệp PDF để lưu các khung hình khung xương
pdf_path = "output.pdf"  # Đường dẫn để lưu tệp PDF
pdf_pages = PdfPages(pdf_path)

# Đọc ảnh từ thư mục
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

frame_count = 0
no_of_frame = 100  # Số frame cần lấy

for image_file in image_files:
    if frame_count >= no_of_frame:
        break

    frame = cv2.imread(image_file)
    if frame is not None:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        lm_list=[]
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]

                lm_list.append([class_id, confidence, box[0], box[1], box[2], box[3]])

            frame = draw_labels_and_boxes(frame, boxes, confidences, class_ids, classes)

            # Lưu khung hình vào tệp PDF
            plt.figure(figsize=(10, 7))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            pdf_pages.savefig()
            plt.close()

        frame_count += 1

# Đóng tệp PDF vừa ghi
pdf_pages.close()

# Ghi vào File CSV
df = pd.DataFrame(lm_list, columns=["class_id", "confidence", "x", "y", "width", "height"])
df.to_csv('output.csv', index=False)
