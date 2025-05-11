import requests
import base64
from io import BytesIO
from PIL import Image

url = "http://192.168.10.139:5000/detect"
image_path = "test.jpg"

# Mở ảnh dưới dạng file để gửi
with open(image_path, 'rb') as image_file:
    files = {'image': image_file}
    # Gửi POST request tới API
    response = requests.post(url, files=files)

# Kiểm tra nếu có lỗi
if response.status_code == 200:
    result = response.json()

    # Lấy thông báo từ kết quả
    closest_object = result.get('closest_object', None)
    if closest_object:
        print("Đối tượng gần nhất:")
        print(f"Label: {closest_object['label']}")
        print(f"Độ tin cậy: {closest_object['confidence']*100:.2f}%")
        print(f"Khoảng cách tới người dùng: {closest_object['distance']:.2f} pixels")
        # In thêm thông báo hướng của vật cản
        print(f"Hướng của vật cản: {closest_object.get('direction', 'Không xác định')}")
    else:
        print("Không tìm thấy đối tượng gần nhất.")

    # Lấy ảnh base64 và giải mã
    img_base64 = result['image']
    img_bytes = base64.b64decode(img_base64)

    # Lưu ảnh ra file
    img = Image.open(BytesIO(img_bytes))
    img.save("output.png")
    print("Ảnh đã được lưu thành công.")
else:
    print(f"Error: {response.status_code}")
