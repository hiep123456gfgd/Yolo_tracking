import cv2
import os
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from yt_dlp import YoutubeDL
from confluent_kafka import Producer


# # Link YouTube
# https://www.youtube.com/watch?v=cB9Fs9UmcRU
# https://www.youtube.com/watch?v=IXBTD4VgFF4

url = "https://youtu.be/Fu3nDsqC1J0"



# Cấu hình yt-dlp để lấy link stream tốt nhất
ydl_opts = {
    'format': 'best',
    'noplaylist': True
}

with YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_path = info_dict.get("url", None)

output_dir = "image"  # Đường dẫn thư mục đã tồn tại
frame_count = 0  # Đếm số lượng khung hình

# Cấu hình Kafka Producer
kafka_config = {
    'bootstrap.servers': 'localhost:9092',  # Địa chỉ Kafka broker
    'client.id': 'image-producer'
}
producer = Producer(kafka_config)

def delivery_report(err, msg):
    if err is not None:
        print(f"Lỗi gửi tin nhắn: {err}")
    else:
        print(f"Tin nhắn đã gửi đến {msg.topic()} [{msg.partition()}] với offset {msg.offset()}")

# Hàm gửi ảnh qua Kafka
def send_image_to_kafka(frame, topic="image-topic"):
    # Mã hóa ảnh thành chuỗi byte
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Gửi ảnh qua Kafka
    producer.produce(topic, value=image_bytes, callback=delivery_report)
    producer.flush()  # Đảm bảo dữ liệu được gửi ngay lập tức


# Config value
# video_path = "data_ext/nguyenhuekomua.mp4"
conf_threshold = 0.5
tracking_classes = [2, 0]  # Danh sách các class cần theo dõi (xe hơi và xe máy)

# Khởi tạo DeepSort
tracker = DeepSort(max_age=30)

# Khởi tạo YOLOv9
device = "cpu"  # "cuda": GPU, "cpu": CPU, "mps:0"
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model = AutoShape(model)

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Quản lý ID cố định cho từng class
id_mappings = {class_id: {} for class_id in tracking_classes}
id_counters = {class_id: 1 for class_id in tracking_classes}

# Danh sách ID đã đếm trong ROI cho từng class
counted_ids = {class_id: set() for class_id in tracking_classes}

# Định nghĩa ROI (tọa độ x, y của góc trên trái và góc dưới phải)
roi_top_left = (0, 150)
roi_bottom_right = (350, 480)

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy FPS của video
frame_interval = int(fps * 10)  # Tính khoảng cách giữa các frame cần xử lý (10 giây)
frame_count = 0  # Đếm số frame


while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:
        continue
    
    if frame_count % frame_interval == 0:  # Chỉ xử lý frame mỗi 10 giây
        # Thay đổi kích thước khung hình
        frame = cv2.resize(frame, (640, 360))
        # Đưa qua model để detect
        results = model(frame)

        detect = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if tracking_classes is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id not in tracking_classes or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật, gán ID bằng DeepSort
        tracks = tracker.update_tracks(detect, frame=frame)

        # Vẽ ROI trên frame
        
        # roi_top_left = (0, 250)
        # roi_bottom_right = (640, 408)

        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)
        class_counts = {class_id: 0 for class_id in tracking_classes}  # Đếm số lượng xe trong từng class

        # Vẽ lên màn hình các khung chữ nhật kèm ID
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id

                # Lấy tọa độ, class_id để vẽ lên hình ảnh
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)

                # Gán ID cố định cho từng class và track_id
                if track_id not in id_mappings[class_id]:
                    id_mappings[class_id][track_id] = id_counters[class_id]
                    id_counters[class_id] += 1  # Tăng bộ đếm ID cho class này

                unique_id = id_mappings[class_id][track_id]
                label = "{}-{}".format(class_names[class_id], unique_id)

                # Kiểm tra xe có trong ROI
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if (roi_top_left[0] <= center_x <= roi_bottom_right[0]) and (roi_top_left[1] <= center_y <= roi_bottom_right[1]):
                    if track_id not in counted_ids[class_id]:
                        counted_ids[class_id].add(track_id)
                    class_counts[class_id] += 1



                # Vẽ bounding box và label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Tính toán số lượng xe và trạng thái
        x_count = class_counts[2]  # Số xe class 2 (xe hơi)
        y_count = class_counts[0]  # Số xe class 0 (xe máy)
        congestion_status = "TRUE"
        status_color = (0, 255, 0)  # Màu xanh lá

        if  y_count  >=  1:
            congestion_status = "False"
            status_color = (0, 0, 255)  # Màu đỏ

        # Hiển thị số lượng xe và trạng thái
        cv2.putText(frame, f"Xe hoi : {x_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Xe may : {y_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Trang thai: {congestion_status}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2) 

        # Lưu khung hình nếu trạng thái tắc nghẽn
        if congestion_status == "False":
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Lưu khung hình tại {frame_filename}")
            frame_count += 1

        # Gửi ảnh qua Kafka
        send_image_to_kafka(frame, topic="traffic-images")
        
        
    frame_count += 1  # Tăng chỉ số frame

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()