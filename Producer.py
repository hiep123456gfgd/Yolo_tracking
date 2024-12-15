import cv2
import base64
import json
from confluent_kafka import Producer

# Cấu hình Kafka Producer
kafka_config = {
    'bootstrap.servers': 'localhost:9092',  # Địa chỉ Kafka broker
    'client.id': 'image-producer'
}
producer = Producer(kafka_config)

# Hàm callback để kiểm tra trạng thái gửi tin nhắn
def delivery_report(err, msg):
    if err is not None:
        print(f"Lỗi gửi tin nhắn: {err}")
    else:
        print(f"Tin nhắn đã gửi đến {msg.topic()} [{msg.partition()}] với offset {msg.offset()}")

# Hàm gửi ảnh qua Kafka (dùng JSON với Base64)
def send_image_to_kafka_with_json(image_path, address, topic="image-topic"):
    try:
        # Đọc ảnh từ file
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

        # Mã hóa ảnh thành chuỗi byte và chuyển đổi sang Base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Tạo payload JSON
        payload = {
            "address": address,
            "image": image_base64
        }

        # Gửi payload dưới dạng JSON
        producer.produce(topic, value=json.dumps(payload).encode('utf-8'), callback=delivery_report)
        producer.flush()
        print(f"Đã gửi ảnh và địa chỉ đến topic: {topic}")

    except Exception as e:
        print(f"Lỗi khi gửi ảnh: {e}")

# Ví dụ: Gửi một ảnh từ file kèm địa chỉ
image_path = "image/frame_0.jpg"
address = "Camera1"
send_image_to_kafka_with_json(image_path, address)
