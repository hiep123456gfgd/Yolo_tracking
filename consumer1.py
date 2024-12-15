import cv2
import base64
import numpy as np  # Đảm bảo import numpy
import json
import time
import os  # Để thao tác với thư mục
from confluent_kafka import Consumer

# Cấu hình Kafka Consumer
kafka_consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'image-consumer-group',
    'auto.offset.reset': 'latest'
}
consumer = Consumer(kafka_consumer_config)
consumer.subscribe(['image-topic'])

# Hàm lưu ảnh vào thư mục theo địa chỉ
def save_image(address, frame):
    # Tạo thư mục theo địa chỉ nếu chưa tồn tại
    directory_path = f"saved_images/{address}"
    os.makedirs(directory_path, exist_ok=True)

    # Tạo tên file ảnh với timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(directory_path, f"image_{timestamp}.jpg")

    # Lưu ảnh dưới dạng .jpg
    cv2.imwrite(file_path, frame)
    print(f"Ảnh đã được lưu vào: {file_path}")

# Hàm tiêu thụ tin nhắn
def consume_images_with_json():
    while True:
        msg = consumer.poll(1.0)  # Đợi tin nhắn (timeout 1 giây)
        if msg is None:
            continue
        if msg.error():
            print(f"Lỗi Consumer: {msg.error()}")
            continue

        try:
            # Giải mã payload JSON
            payload = json.loads(msg.value().decode('utf-8'))
            address = payload.get("address")
            image_base64 = payload.get("image")

            # Giải mã Base64 thành chuỗi byte
            image_bytes = base64.b64decode(image_base64)

            # Chuyển đổi chuỗi byte thành mảng NumPy
            frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Hiển thị ảnh và thông tin
            print(f"Đã nhận ảnh từ địa chỉ: {address}")
            cv2.imshow("Received Frame", frame)

            # Lưu ảnh vào thư mục theo địa chỉ
            save_image(address, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Lỗi xử lý dữ liệu: {e}")

    consumer.close()
    cv2.destroyAllWindows()

# Chạy Consumer
consume_images_with_json()
