#  ImprovedResSENet - Phân loại ảnh bằng Deep Learning với dưới 100K tham số

Kiến trúc CNN cho bài toán phân loại ảnh 4 lớp, đạt 99.04% accuracy với 64,388 tham số . Mô hình kết hợp ba kỹ thuật : Depthwise Separable Convolution, Residual Connection và Squeeze-and-Excitation Attention.
Kết quả

### Mô hình cuối cùng

| Chỉ số | Giá trị |
|---|---|
| Validation Accuracy | **99.04%** |
| Macro F1-score | **0.9909** |
| Tổng tham số | **64,388** |
| Epoch tốt nhất | 40 |

### Chi tiết theo từng lớp

| Lớp | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Class01 | 98.83% | 98.25% | 98.54% | 343 |
| Class02 | **100.00%** | 99.62% | 99.81% | 262 |
| Class03 | 99.47% | **100.00%** | 99.74% | 376 |
| Class04 | 98.13% | 98.40% | 98.27% | 374 |
| **Macro Avg** | **99.11%** | **99.07%** | **99.09%** | 1,355 |

Toàn bộ 4 lớp đều đạt Precision và Recall trên 98%, cho thấy mô hình hoạt động đồng đều, không bỏ sót lớp nào.

### So sánh giữa các biến thể kiến trúc

Kiến trúc đề xuất

```
Input (256×256×3)
      ↓
Stem: Conv2d 3×3 → BatchNorm → ReLU        (24 kênh)
      ↓
Stage 1: DSConv → SE-Attention → Residual(+) → Dropout → MaxPool   (128×128×48)
      ↓
Stage 2: DSConv → SE-Attention → Residual(+) → Dropout → MaxPool   (64×64×80)
      ↓
Stage 3: DSConv → SE-Attention → Residual(+) → Dropout → MaxPool   (32×32×112)
      ↓
Stage 4: DSConv → SE-Attention → Residual(+) → Dropout → MaxPool   (16×16×112)
      ↓
Global Average Pooling → FC(112→96) → ReLU → Dropout → FC(96→4)
      ↓
Output: 4 raw logits
```

Số kênh tăng dần `24 → 48 → 80 → 112` trong khi kích thước không gian giảm dần `256 → 16`, theo đúng nguyên lý thu hẹp không gian và mở rộng kênh đặc trưng.

### Ba kỹ thuật cải tiến và lý do lựa chọn

**1. Depthwise Separable Convolution** 
Tách một phép tích chập thường thành hai bước: depthwise (lọc riêng từng kênh) và pointwise (trộn kênh bằng kernel 1×1).

Ví dụ tại Stage 1 (24 → 48 kênh):
- Tích chập thường: `3×3×24×48` = **10,368** tham số
- Depthwise separable: `(3×3×24) + (24×48)` = 216 + 1,152 = **1,368** tham số
-> Giúp giảm số lượng tham số khoảng 7.6 lần**


**2. Residual Connection** - giúp mạng sâu hội tụ ổn định

Công thức `y = F(x) + P(x)`, trong đó `P(x)` là nhánh projection dùng Conv 1×1 để đồng bộ số kênh khi input và output khác nhau. Gradient truyền thẳng qua kết nối tắt, hạn chế hiện tượng vanishing gradient.

**3. SE-Attention (Squeeze-and-Excitation)** - tăng khả năng phân biệt lớp khó

Cơ chế chú ý theo kênh gồm 2 bước:
- **Squeeze:** Global Average Pooling nén mỗi kênh thành 1 giá trị
- **Excitation:** hai tầng FC với ReLU và Sigmoid học ra hệ số quan trọng cho từng kênh, rồi nhân ngược lại vào feature map

-> Giúp mô hình tự học cách nhấn mạnh các kênh phản ánh đặc trưng phân biệt (ví dụ cường độ và hướng lan của vệt loang bệnh), đồng thời tắt các kênh chứa nhiễu nền.

##  Dataset

- **Tổng số ảnh:** 6,776 ảnh, chia thành 4 lớp
- **Phân bố:** Class01 (1,715), Class02 (1,310), Class03 (1,880), Class04 (1,871) — tương đối cân bằng
- **Chia dữ liệu:** 80% train (5,421 ảnh) và 20% validation (1,355 ảnh), với seed cố định

**Đặc điểm từng lớp** (quan sát trực tiếp):

| Lớp | Đặc trưng hình ảnh | Độ khó |
|---|---|---|
| Class01 | Vệt loang khô vàng đến nâu nhạt, kéo dài dọc gân lá | Trung bình |
| Class02 | Đốm tròn nhỏ nâu cam, viền sắc nét, phân bố đều | Dễ |
| Class03 | Lá khỏe mạnh, bề mặt xanh đồng đều | Dễ nhất |
| Class04 | **Đặc trưng hỗn hợp**, vừa có vệt loang vừa có đốm tròn | Khó nhất |

---

##  Cấu hình huấn luyện

| Siêu tham số | Giá trị |
|---|---|
| Loss function | CrossEntropyLoss (label_smoothing = 0.05) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 4) |
| Batch size | 32 |
| Epochs | 40 |
| Random seed | 42 |

**Data augmentation** (chỉ áp dụng cho tập train): RandomHorizontalFlip, RandomRotation (±12°), ColorJitter, RandomAffine. Tập validation chỉ Resize, ToTensor và Normalize để đánh giá khách quan.

**Tính tái lập:** seed 42 được cố định cho toàn bộ nguồn ngẫu nhiên (Python `random`, `PYTHONHASHSEED`, NumPy, PyTorch CPU/GPU) và `cudnn.deterministic = True`. Việc chia train/val dùng Generator riêng có seed cố định, đảm bảo cả ba biến thể được đánh giá trên **đúng cùng một tập ảnh**.


---

##  Phân tích lỗi

Trên 1,355 ảnh validation, mô hình chỉ sai **13 ảnh** (tỷ lệ lỗi 0.96%):

| Nhãn thật | Đoán đúng | Đoán nhầm |
|---|---|---|
| Class01 (343) | 337 | 6 ảnh → Class04 |
| Class02 (262) | 261 | 1 ảnh → Class04 |
| Class03 (376) | **376** | 0 ảnh |
| Class04 (374) | 368 | 4 ảnh → Class01, 2 ảnh → Class03 |

**Nhận định:** hầu hết lỗi tập trung vào cặp **Class01 ↔ Class04**. Điều này hoàn toàn khớp với đặc điểm dữ liệu: Class04 mang **đặc trưng hỗn hợp**, vừa có vệt loang giống Class01 vừa có đốm tròn giống Class02. Khi vùng vệt loang chiếm ưu thế trong một ảnh Class04, mô hình dễ nhầm sang Class01.

**Về giá trị loss:** validation loss hội tụ quanh 0.23 đến 0.24 chứ không tiến về 0, và đây là điều bình thườngdo dùng label smoothing. Giới hạn dưới lý thuyết của loss trong trường hợp này là:

```
H(p) = -(0.9625·ln 0.9625 + 3 × 0.0125·ln 0.0125) ≈ 0.2011
```

