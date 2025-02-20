1. Triển khai mô hình Image Captioning sinh mô tả tiếng Việt cho hình ảnh, hỗ trợ người khiếm thị di chuyển thuận tiện hơn bằng Google Colab.

1.2. Các thành phần chính

Dữ liệu: Sử dụng hình ảnh từ các bộ dữ liệu có sẵn như MS COCO, Flickr30k.

Mô hình:

Encoder: Mô hình CNN (ResNet-50) trích xuất đặc trưng hình ảnh.

Decoder: Mô hình LSTM dựa trên đặc trưng để sinh caption.

Huấn luyện: Chạy trên Google Colab Free.

Dự đoán: Sinh caption tiếng Việt cho hình ảnh.

2. Chuẩn bị dữ liệu

2.1. Các thành phần cần thiết

Hình ảnh từ bộ dữ liệu có sẵn.

Caption tiếng Việt: Dịch caption từ tiếng Anh (nếu cần).

Từ điển (Vocabulary): Danh sách từ vựng trong caption.

2.2. Tạo dữ liệu caption tiếng Việt

Sử dụng Google Translate API hoặc tự dịch bằng tay.

Xây dựng từ điển cho caption.

2.3. Chia tập dữ liệu

Train: 80%.

Validation: 10%.

Test: 10%.

3. Xây dựng mô hình

3.1. Encoder - Trích xuất đặc trưng

Sử dụng ResNet-50 (loại bỏ Fully Connected Layer).

3.2. Decoder - Sinh caption

Sử dụng LSTM.

Nhận đầu vào là đặc trưng từ Encoder + caption trước đó.

3.3. Tích hợp Encoder và Decoder

Kết hợp ResNet-50 làm Encoder và LSTM làm Decoder.
