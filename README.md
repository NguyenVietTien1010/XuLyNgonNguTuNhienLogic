
# 🤖 Trợ Lý Logic – Dự đoán hành động từ thời tiết và cảm xúc

Ứng dụng Streamlit sử dụng kỹ thuật **xử lý ngôn ngữ tự nhiên (NLP)** kết hợp với **mạng nơ-ron nhân tạo (MLP)** để dự đoán hành động của người dùng (đi chơi / nghỉ ngơi) dựa trên mô tả ngắn về **thời tiết** và **tâm trạng**.

---

## 🧠 Tính năng chính

- Nhập mô tả tình huống bằng tiếng Việt
- Mô hình MLP học từ đặc trưng đã mã hóa
- Phân tích biểu thức logic (AND, OR, NOT, XOR)
- Giải thích quyết định dự đoán theo cả logic và học máy
- Giao diện tương tác trực quan, dễ sử dụng

---

## 🗃️ Dữ liệu

Dữ liệu gồm các mô tả về thời tiết, tâm trạng và hành động mong muốn. Ví dụ:

| weather  | mood       | action     | explanation                        |
|----------|------------|------------|------------------------------------|
| nắng     | rất muốn   | đi chơi    | Trời đẹp, tâm trạng tốt...         |
| âm u     | mệt        | nghỉ ngơi  | Không khí u ám, cơ thể mệt mỏi…    |

---

## 🧮 Mô hình học máy

- Mô hình: Multi-Layer Perceptron (MLP)
- Đầu vào: thời tiết, tâm trạng (mã hóa), đặc trưng logic
- Tăng cường với các biến như `is_weather_good`, `logic_and`, `logic_xor`...
- Độ chính xác ~94–97% trên tập test
- Trọng số được lưu ở `mlp_model.pth`

---

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run main.py
# Hoặc bản có biểu đồ loss:
streamlit run main2.py
```

Sau đó truy cập `http://localhost:8501` trên trình duyệt.

---

## 📋 Cấu trúc chính

| File               | Mô tả                                     |
|--------------------|--------------------------------------------|
| `main.py`          | Giao diện chính, phân tích logic sâu       |
| `main2.py`         | Giao diện + biểu đồ loss                   |
| `data.csv`         | Dữ liệu gốc                                |
| `data2.csv`        | Dữ liệu lọc (chỉ gồm đi chơi / nghỉ ngơi)  |
| `mlp_model.pth`    | File lưu mô hình đã huấn luyện             |
| `losses.pkl`       | File lưu loss các epoch (main2.py)         |
| `requirements.txt` | Thư viện cần thiết                         |

---

## 🧠 Các đặc trưng logic được sử dụng

| Biến            | Ý nghĩa                                      |
|-----------------|-----------------------------------------------|
| is_weather_good | Trời có đẹp hay không (nắng, đẹp, nắng nhẹ)   |
| is_mood_positive| Tâm trạng tích cực (vui, muốn, hứng thú)     |
| logic_and       | Cả trời đẹp và tâm trạng tốt                 |
| logic_or        | Một trong hai điều kiện đúng                 |
| logic_xor       | Chỉ một điều kiện đúng (cân nhắc)            |

---

## 📈 Biểu diễn mô hình

Ứng dụng hiển thị **cấu trúc mạng MLP bằng Graphviz**, giúp người học dễ hình dung kiến trúc học sâu.

---

## 👤 Tác giả

Nguyễn Viết Tiến  


