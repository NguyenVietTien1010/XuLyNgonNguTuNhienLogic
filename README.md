# 🤖 Trợ Lý Logic – Dự đoán hành động từ thời tiết và cảm xúc

Ứng dụng này sử dụng mạng nơ-ron nhân tạo (MLP) kết hợp với biểu diễn logic để **dự đoán hành động của con người** (như đi chơi hay nghỉ ngơi) dựa vào **thời tiết** và **tâm trạng**. Giao diện được xây dựng bằng **Streamlit**, hỗ trợ nhập ngôn ngữ tự nhiên bằng tiếng Việt.

---

## 🧠 Tính năng chính

- ✅ Dự đoán hành động dựa trên thời tiết và cảm xúc.
- ✅ Phân tích logic với các phép AND, OR, NOT, XOR.
- ✅ Giao diện chat tương tác và giải thích quyết định.
- ✅ Biểu đồ thống kê dữ liệu và loss qua từng epoch.
- ✅ Mô hình huấn luyện bằng PyTorch, có khả năng lưu lại trọng số và tái sử dụng.

---

## 🗃️ Cấu trúc dữ liệu

Các tệp CSV chứa các cột chính như:

| weather  | mood         | action     | explanation                     |
|----------|--------------|------------|----------------------------------|
| nắng     | rất muốn     | đi chơi    | Trời đẹp, tâm trạng tốt...       |
| âm u     | mệt          | nghỉ ngơi  | Không khí u ám, cơ thể mệt mỏi… |

Mã nguồn sẽ mã hóa các cột thành số để huấn luyện mô hình MLP.

---

## 🧮 Mô hình học máy

- Sử dụng **Multi-Layer Perceptron (MLP)** với nhiều lớp ẩn.
- Hàm mất mát: `CrossEntropyLoss` (có trọng số nếu mất cân bằng).
- Tối ưu hóa: `Adam`
- Huấn luyện trong 500 epoch (có lưu `losses.pkl`).
- Accuracy hiện tại trên tập test ~ **94-97%** tùy phiên bản dữ liệu.

---

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt môi trường

# 🤖 Trợ Lý Logic – Dự đoán hành động từ thời tiết và cảm xúc

Ứng dụng này sử dụng mạng nơ-ron nhân tạo (MLP) kết hợp với biểu diễn logic để **dự đoán hành động của con người** (như đi chơi hay nghỉ ngơi) dựa vào **thời tiết** và **tâm trạng**. Giao diện được xây dựng bằng **Streamlit**, hỗ trợ nhập ngôn ngữ tự nhiên bằng tiếng Việt.

---

## 🧠 Tính năng chính

- ✅ Dự đoán hành động dựa trên thời tiết và cảm xúc.
- ✅ Phân tích logic với các phép AND, OR, NOT, XOR.
- ✅ Giao diện chat tương tác và giải thích quyết định.
- ✅ Biểu đồ thống kê dữ liệu và loss qua từng epoch.
- ✅ Mô hình huấn luyện bằng PyTorch, có khả năng lưu lại trọng số và tái sử dụng.

---

## 🗃️ Cấu trúc dữ liệu

Các tệp CSV chứa các cột chính như:

| weather  | mood         | action     | explanation                     |
|----------|--------------|------------|----------------------------------|
| nắng     | rất muốn     | đi chơi    | Trời đẹp, tâm trạng tốt...       |
| âm u     | mệt          | nghỉ ngơi  | Không khí u ám, cơ thể mệt mỏi… |

Mã nguồn sẽ mã hóa các cột thành số để huấn luyện mô hình MLP.

---

## 🧮 Mô hình học máy

- Sử dụng **Multi-Layer Perceptron (MLP)** với nhiều lớp ẩn.
- Hàm mất mát: `CrossEntropyLoss` (có trọng số nếu mất cân bằng).
- Tối ưu hóa: `Adam`
- Huấn luyện trong 500 epoch (có lưu `losses.pkl`).
- Accuracy hiện tại trên tập test ~ **94-97%** tùy phiên bản dữ liệu.

---

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt môi trường

🚀 Cách chạy ứng dụng
- ✅ Yêu cầu môi trường
Python 3.8 trở lên
Đã cài pip
Hệ điều hành: Windows / macOS / Linux


