import cv2
import numpy as np
import os
import time
import json  # Dùng để xử lý JSON
import base64  # Giải mã Base64
from confluent_kafka import Consumer

# Cấu hình Kafka Consumer
kafka_consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'image-consumer-group',
    'auto.offset.reset': 'latest'
}
consumer = Consumer(kafka_consumer_config)

# Đăng ký topic
consumer.subscribe(['traffic-images22'])

# Hàm lưu ảnh dưới dạng .jpg với tên file động
def save_image_as_jpg(address, image_base64):
    """
    Lưu ảnh nhận được vào file dưới dạng .jpg với tên file động, phân loại theo địa chỉ.
    """
    # Tạo thư mục dựa trên địa chỉ
    directory_path = f'saved_images/{address}'
    os.makedirs(directory_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    # Tạo tên file dựa trên timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(directory_path, f'image_{timestamp}.jpg')

    try:
        # Giải mã Base64 sang bytes
        image_data = base64.b64decode(image_base64)

        # Chuyển bytes thành ảnh
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Lưu ảnh dưới dạng .jpg
            cv2.imwrite(file_path, frame)
            print(f"Ảnh đã được lưu vào: {file_path}")
        else:
            print("Dữ liệu không phải là ảnh hợp lệ.")
    except Exception as e:
        print(f"Lỗi khi giải mã hoặc lưu ảnh: {e}")

# Hàm tiêu thụ và xử lý tin nhắn Kafka
def consume_images():
    while True:
        msg = consumer.poll(1.0)  # Đợi tin nhắn (timeout 1 giây)
        if msg is None:
            continue
        if msg.error():
            print(f"Lỗi Consumer: {msg.error()}")
            continue

        # Giải mã tin nhắn dạng bytes
        try:
            # Chuyển từ JSON string thành dictionary
            message = json.loads(msg.value().decode('utf-8'))  # Dùng json.loads thay cho eval hoặc ast
            address = message.get('address')
            image_base64 = message.get('image')

            if address and image_base64:
                # Lưu ảnh vào file dưới dạng .jpg với tên động theo địa chỉ
                save_image_as_jpg(address, image_base64)
            else:
                print("Dữ liệu không đầy đủ hoặc không hợp lệ.")

        except Exception as e:
            print(f"Lỗi xử lý dữ liệu: {e}")

    consumer.close()

# Chạy Consumer
consume_images()
