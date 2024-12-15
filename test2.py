import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from yt_dlp import YoutubeDL

# # Link YouTube
# https://www.youtube.com/watch?v=cB9Fs9UmcRU
# https://www.youtube.com/watch?v=IXBTD4VgFF4

url = "https://youtu.be/IXBTD4VgFF4"



# Cấu hình yt-dlp để lấy link stream tốt nhất
ydl_opts = {
    'format': 'best',
    'noplaylist': True
}

with YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_path = info_dict.get("url", None)



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
# Định nghĩa ROI ngũ giác
roi_pentagon = np.array([ [300, 360],[640, 360], [640, 190], [380, 124], [110, 124]], np.int32)
roi_pentagon = roi_pentagon.reshape((-1, 1, 2))  # Định dạng cho OpenCV


# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:
        continue

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

    cv2.polylines(frame, [roi_pentagon], isClosed=True, color=(0, 255, 0), thickness=2)
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
            if cv2.pointPolygonTest(roi_pentagon, (center_x, center_y), False) >= 0:
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

    if x_count + y_count / 4 >= 8 :
        congestion_status = "False"
        status_color = (0, 0, 255)  # Màu đỏ

    # Hiển thị số lượng xe và trạng thái
    cv2.putText(frame, f"Xe hoi : {x_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Xe may : {y_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Trang thai: {congestion_status}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)


    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()