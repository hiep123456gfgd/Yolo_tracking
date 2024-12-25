import cv2
import os
import torch
import numpy as np
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from yt_dlp import YoutubeDL
from confluent_kafka import Producer
import base64
import json
from dotenv import load_dotenv
from datetime import datetime
import cloudinary
import cloudinary.uploader

load_dotenv()

video_configs = [
    {
        "camera_id": "676bfabbedf62fc4a83b5027",
        "url": "https://youtu.be/Fu3nDsqC1J0",
        "roi": [
            [0, 360],
            [200, 360],
            [410, 120],
            [140, 120],
            [0, 150]
        ],
        "image": "image1",
        "latitude": 16.074108898317647,
        "longitude":108.215779060517,
        
    },
    {
        "camera_id": "676bfabcedf62fc4a83b5028",
        "url": "https://youtu.be/IXBTD4VgFF4",
        "roi": [
            [300, 360],
            [640, 360],
            [640, 190],
            [380, 124],
            [110, 124]
        ],
        "image": "image2",
        "latitude": 16.072839621050942,
        "longitude": 108.21654270831574,
    }
]

# Cấu hình yt-dlp để lấy link stream tốt nhất
ydl_opts = {
    'format': 'best',
    'noplaylist': True
}
# Cấu hình Kafka Producer
kafka_config = {
    'bootstrap.servers': os.getenv('BOOTSTRAP_SERVERS'),  # Địa chỉ Kafka broker từ biến môi trường
    'security.protocol': 'SASL_SSL',                    # Kết nối qua SSL
    'sasl.mechanism': 'PLAIN',                          # Cơ chế SASL (Plain)
    'sasl.username': os.getenv('SASL_USERNAME'),        # Tên người dùng SASL
    'sasl.password': os.getenv('SASL_PASSWORD'),        # Mật khẩu SASL
    'client.id': 'image-producer'                       # ID client
}


cloudinary.config(
    cloud_name="dvrisaqgy",
    api_key="816794745326251",
    api_secret=os.getenv("CLOUDINARY_API_SECRET")  # Đọc từ biến môi trường
)

# Khởi tạo Producer
producer = Producer(kafka_config)


def delivery_report(err, msg):
    if err is not None:
        print(f"Lỗi gửi tin nhắn: {err}")
    else:
        print(f"Tin nhắn đã gửi đến {msg.topic()} [{msg.partition()}] với offset {msg.offset()}")

# Hàm gửi ảnh và thông tin address qua Kafka
def send_image_to_kafka(frame,conges, video_config, img, topic="python-topic"):
    message = {
        "camera_id": video_config["camera_id"],
        "type": "camera report",
        "typeReport": "TRAFFIC_JAM",
        "congestionLevel": "HEAVY_CONGESTION",
        "description": "Traffic Jam in camera",
        "trafficVolume": conges,
        "longitude": video_config["longitude"],
        "latitude": video_config["latitude"],
        "timestamp": datetime.now().isoformat(),
        "img": img,
    }

    serialized_message = json.dumps(message)
    producer.produce(topic, value=serialized_message, callback=delivery_report)
    producer.flush()  # Đảm bảo dữ liệu được gửi ngay lập tức

# Cấu hình YOLOv9 và DeepSort
conf_threshold = 0.5
tracking_classes = [2, 0]  # Theo dõi xe hơi và xe máy
device = "cpu"  # Chuyển sang "cuda" nếu có GPU
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model = AutoShape(model)

with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(class_names), 3))


def process_stream(video_config, stream_id):
    # Khởi tạo DeepSort
    tracker = DeepSort(max_age=30)
    
    local_frame_count = 0  # Đếm số lượng khung hình

    # Lấy URL stream từ YouTube
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_config["url"], download=False)
        video_path = info_dict.get("url", None)

    # Khởi tạo VideoCapture
    cap = cv2.VideoCapture(video_path)
    roi_pentagon = np.array(video_config["roi"], np.int32).reshape((-1, 1, 2))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy FPS của video
    frame_interval = int(fps * 10)  # Tính khoảng cách giữa các frame cần xử lý (10 giây)
    frame_count = 0  # Đếm số frame
    

    # Quản lý ID
    id_mappings = {class_id: {} for class_id in tracking_classes}
    id_counters = {class_id: 1 for class_id in tracking_classes}
    counted_ids = {class_id: set() for class_id in tracking_classes}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))    
        
        if frame_count % frame_interval == 0:  # Chỉ xử lý frame mỗi 10 giây
            frame = cv2.resize(frame, (640, 360))
            results = model(frame)

            detect = []
            for detect_object in results.pred[0]:
                label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)

                if class_id not in tracking_classes or confidence < conf_threshold:
                    continue

                detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

            # Cập nhật DeepSort
            tracks = tracker.update_tracks(detect, frame=frame)

            # Vẽ ROI
            cv2.polylines(frame, [roi_pentagon], isClosed=True, color=(0, 255, 0), thickness=2)
            class_counts = {class_id: 0 for class_id in tracking_classes}

            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = colors[class_id]
                    B, G, R = map(int, color)

                    # Gán ID cố định
                    if track_id not in id_mappings[class_id]:
                        id_mappings[class_id][track_id] = id_counters[class_id]
                        id_counters[class_id] += 1

                    unique_id = id_mappings[class_id][track_id]
                    label = "{}-{}".format(class_names[class_id], unique_id)

                    # Kiểm tra ROI
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    if cv2.pointPolygonTest(roi_pentagon, (center_x, center_y), False) >= 0:
                        if track_id not in counted_ids[class_id]:
                            counted_ids[class_id].add(track_id)
                        class_counts[class_id] += 1

                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Hiển thị thông tin
            # Tính toán số lượng xe và trạng thái
            x_count = class_counts[2]  # Số xe class 2 (xe hơi)
            y_count = class_counts[0]  # Số xe class 0 (xe máy)
            congestion_status = "TRUE"
            status_color = (0, 255, 0)  # Màu xanh lá

            if x_count + y_count / 4 >= 8 :
                congestion_status = "False"
                status_color = (0, 0, 255)  # Màu đỏ

            # Hiển thị số lượng xe và trạng thái
            cv2.putText(frame, f"Xe hoi : {x_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Xe may : {y_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Trang thai: {congestion_status}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Lưu khung hình nếu trạng thái tắc nghẽn
            if congestion_status == "False":
                # Gửi ảnh và thông tin địa chỉ qua Kafka
                conges = x_count + y_count

                # Sử dụng thông tin từ video_config để lưu vào thư mục tương ứng
                image_dir = video_config["image"]
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)  # Tạo thư mục nếu chưa tồn tại

                # Lưu khung hình vào thư mục tương ứng
                frame_filename = os.path.join(image_dir, f"stream_{stream_id}_frame_{local_frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Lưu khung hình tại {frame_filename}")
                # Lưu frame vào Cloudinary
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    upload_result = cloudinary.uploader.upload(
                        buffer.tobytes(),
                        folder=f"stream_{stream_id}",
                        public_id=f"frame_{local_frame_count}",
                        resource_type="image"
                    )
                    img_url = upload_result.get("url")
                    print(f"Uploaded to Cloudinary: {img_url}")
                except Exception as e:
                    print(f"Error uploading to Cloudinary: {e}")
                    img_url = None  # Đặt URL ảnh là None nếu upload thất bại

                # Chỉ gửi thông tin qua Kafka nếu upload thành công
                if img_url:
                    send_image_to_kafka(frame, conges, video_config, img_url, topic="python-topic")

                local_frame_count += 1
        frame_count += 1  # Tăng chỉ số frame
            
        # Hiển thị frame
        cv2.imshow(f"Stream {stream_id}", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Chạy nhiều luồng stream
threads = []
for idx, video_config in enumerate(video_configs):
    thread = threading.Thread(target=process_stream, args=(video_config, idx + 1))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
