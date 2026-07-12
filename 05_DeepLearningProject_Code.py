# ======================================================================
# ĐỒ ÁN CUỐI KỲ - MÔN CƠ SỞ VÀ ỨNG DỤNG AI
# Đề tài: Phân loại ảnh tổng quát (General Image Classification)
# Kiến trúc đề xuất: ImprovedResSENet
# Nhóm: 05
# ======================================================================
#
# Mô hình do nhóm tự thiết kế, không sao chép kiến trúc baseline mẫu
# (BasicCNN) mà giảng viên cung cấp. Kiến trúc kết hợp 3 kỹ thuật cải
# tiến hiện đại (đúng yêu cầu mục 6.3 của đề bài):
#
#   1) Depthwise Separable Convolution: tách 1 phép tích chập thông
#      thường thành 2 bước nhỏ (depthwise + pointwise), giúp giảm đáng
#      kể số lượng tham số cần huấn luyện.
#   2) Residual Connection (kết nối tắt): cộng trực tiếp đầu vào gốc
#      của mỗi khối vào đầu ra, giúp gradient lan truyền tốt hơn qua
#      các tầng sâu, hạn chế hiện tượng suy giảm gradient, giúp huấn
#      luyện hội tụ ổn định hơn.
#   3) SE-Attention (Squeeze-and-Excitation): cơ chế chú ý theo kênh,
#      giúp mô hình tự học cách tập trung vào những kênh đặc trưng
#      quan trọng nhất cho việc phân loại (ví dụ phân biệt vệt loang
#      khô và đốm tròn giữa các lớp lá bệnh khác nhau trong dataset).
#
# ======================================================================
# HƯỚNG DẪN CHẠY TRÊN KAGGLE:
#
#   1. Add Input dataset "Data_CLC_Classification" (tác giả huynhthethien)
#      vào notebook trước khi chạy.
#   2. Bật GPU: Settings > Accelerator > GPU T4 x2, để quá trình huấn
#      luyện không bị chậm.
#   3. Cell huấn luyện (Cell 5) là cell tốn thời gian nhất, khoảng
#      25-30 phút trên GPU T4 - không tắt tab khi đang chạy.
# ======================================================================


# ==========================================================================
# CELL 1 - CẤU HÌNH HẰNG SỐ, THIẾT BỊ VÀ CỐ ĐỊNH SEED
# ==========================================================================

import os                              # thao tác biến môi trường, đường dẫn file
import random                          # bộ sinh số ngẫu nhiên của Python core
from pathlib import Path               # xử lý đường dẫn file/thư mục

import numpy as np                     # xử lý mảng số, dùng cho confusion matrix
import torch                           # framework chính để xây dựng và huấn luyện mạng
import torch.nn as nn                  # các lớp mạng nơ-ron có sẵn (Conv2d, Linear...)
import torch.optim as optim            # các thuật toán tối ưu (AdamW...)
from torch.utils.data import DataLoader, random_split   # chia batch và chia dataset
from torchvision import datasets, transforms             # đọc ảnh và tiền xử lý ảnh

import matplotlib
matplotlib.use("Agg")  # tắt hiển thị giao diện đồ thị, cần thiết khi chạy trên Kaggle
import matplotlib.pyplot as plt                                   # vẽ đồ thị loss/accuracy
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # tính chỉ số đánh giá
import seaborn as sns                                              # vẽ confusion matrix dạng heatmap

# Cố định seed cho toàn bộ các nguồn sinh số ngẫu nhiên (Python, NumPy,
# PyTorch CPU/GPU) để đảm bảo tính tái lập của kết quả, theo đúng yêu
# cầu mục 7.4 của đề bài.
SEED = 42

def set_seed(seed):
    random.seed(seed)                                  # cố định seed cho Python core
    os.environ["PYTHONHASHSEED"] = str(seed)            # cố định seed cho hashing của hệ điều hành
    np.random.seed(seed)                                 # cố định seed cho NumPy
    torch.manual_seed(seed)                                # cố định seed cho PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)                         # cố định seed cho PyTorch (toàn bộ GPU)
    torch.backends.cudnn.deterministic = True                 # bắt cuDNN dùng thuật toán xác định
    torch.backends.cudnn.benchmark = False                      # tắt tự động chọn thuật toán nhanh nhất (vì có thể không xác định)

set_seed(SEED)   # gọi hàm ngay để áp dụng seed trước khi làm bất kỳ việc gì có tính ngẫu nhiên


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện của mô hình, dùng để kiểm
    tra điều kiện bắt buộc phải nhỏ hơn 100,000 theo mục 6.4 đề bài."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Đường dẫn dataset trên Kaggle (đã kiểm tra khớp với dataset thật bằng
# os.walk('/kaggle/input') trước khi chạy chính thức)
DATA_DIR = "/kaggle/input/datasets/huynhthethien/data-clc-classification"

# Cấu hình hằng số chính của mô hình
IMAGE_SIZE = 256          # KHÔNG ĐƯỢC THAY ĐỔI - quy định bắt buộc mục 14.2.1
NUM_CLASSES = 4           # 4 lớp: Class01, Class02, Class03, Class04
BATCH_SIZE = 32           # số ảnh đưa vào mạng mỗi lần lặp
EPOCHS = 40               # tổng số vòng lặp huấn luyện qua toàn bộ tập train
LEARNING_RATE = 1e-3      # tốc độ học ban đầu của optimizer AdamW
WEIGHT_DECAY = 1e-4       # hệ số weight decay (L2 regularization), giúp chống overfitting
VALID_RATIO = 0.2         # tỷ lệ trích tập validation để tự đánh giá nội bộ trong lúc làm

GroupID = "05"            # số thứ tự nhóm, dùng để đặt tên file model khi export

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # tự chọn GPU nếu có, không thì dùng CPU
print("Thiết bị đang sử dụng để huấn luyện:", DEVICE)   # in ra để kiểm tra đã nhận đúng GPU chưa

# ==========================================================================
# CELL 2 - PIPELINE TIỀN XỬ LÝ DỮ LIỆU
# ==========================================================================
# Định nghĩa các bước biến đổi ảnh trước khi đưa vào mạng. Mọi phép biến
# đổi chỉ chạy động trong bộ nhớ (RAM) mỗi lần đọc ảnh, không ghi đè hay
# tạo thêm file ảnh vào dataset gốc, đúng quy định mục 5 của đề bài.

# Transform cho tập TRAIN: có augmentation để tăng tính đa dạng của dữ
# liệu huấn luyện, giúp mô hình tổng quát hóa tốt hơn và hạn chế hiện
# tượng overfitting.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),               # đưa ảnh về đúng kích thước chuẩn 256x256
    transforms.RandomHorizontalFlip(p=0.5),                     # lật ngang ảnh với xác suất 50%
    transforms.RandomRotation(degrees=12),                      # xoay ảnh ngẫu nhiên trong khoảng +-12 độ
    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),  # thay đổi nhẹ độ sáng, tương phản, độ bão hòa màu
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # dịch ảnh nhẹ theo chiều ngang và dọc
    transforms.ToTensor(),                                      # chuyển ảnh sang dạng tensor để tính toán trên GPU
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # chuẩn hóa giá trị pixel theo mean/std chuẩn ImageNet
])

# Transform cho tập VALIDATION: không augmentation, chỉ resize và chuẩn
# hóa, để đảm bảo đánh giá mô hình trên dữ liệu khách quan, không bị
# biến đổi giả tạo.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                # đưa ảnh về đúng kích thước chuẩn 256x256
    transforms.ToTensor(),                                      # chuyển ảnh sang dạng tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # chuẩn hóa giá trị pixel, dùng đúng mean/std như tập train
])

# ==========================================================================
# CELL 3 - NẠP DATASET VÀ CHIA TỶ LỆ TRAIN/VALIDATION
# ==========================================================================
# Đọc toàn bộ ảnh từ thư mục dataset (Class01-04), chia thành 80% để
# huấn luyện và 20% để tự đánh giá nội bộ trong quá trình làm. Tỷ lệ này
# chỉ phục vụ việc tự kiểm tra của nhóm, không phải kết quả chấm điểm
# chính thức (giảng viên đánh giá trên tập test ẩn riêng).

data_dir = Path(DATA_DIR)                          # chuyển đường dẫn dataset sang dạng Path để xử lý
if not data_dir.exists():                            # kiểm tra thư mục dataset có thực sự tồn tại không
    raise FileNotFoundError(f"DATA_DIR không tồn tại: {data_dir}. "
                             f"Hãy kiểm tra lại đường dẫn dataset trên Kaggle.")

# ImageFolder tự động đọc ảnh từ các thư mục con và gán nhãn theo đúng
# tên thư mục chứa ảnh, không cần gán nhãn thủ công.
full_train = datasets.ImageFolder(root=data_dir, transform=train_transform)  # đọc dataset, áp dụng transform có augmentation
full_val = datasets.ImageFolder(root=data_dir, transform=val_transform)      # đọc lại dataset, áp dụng transform không augmentation
class_names = full_train.classes                                              # lấy danh sách tên các lớp từ tên thư mục

val_size = int(len(full_train) * VALID_RATIO)        # tính số ảnh dành cho tập validation (20%)
train_size = len(full_train) - val_size              # số ảnh còn lại dành cho tập train

# Cố định seed cho generator chia dữ liệu, đảm bảo cách chia train/val
# giống nhau ở mọi lần chạy lại, phục vụ tính tái lập theo mục 7.4 đề bài.
split_generator = torch.Generator().manual_seed(SEED)   # tạo bộ sinh số ngẫu nhiên riêng, cố định theo SEED
train_indices, val_indices = random_split(
    range(len(full_train)), [train_size, val_size], generator=split_generator,
)   # chia ngẫu nhiên các vị trí ảnh thành 2 phần theo đúng tỷ lệ đã tính

train_dataset = torch.utils.data.Subset(full_train, train_indices.indices)  # trích đúng ảnh train theo vị trí vừa chia
val_dataset = torch.utils.data.Subset(full_val, val_indices.indices)        # trích đúng ảnh validation theo vị trí vừa chia

# DataLoader chia dataset thành từng batch nhỏ để đưa vào mô hình tuần
# tự, tránh phải nạp toàn bộ ảnh vào bộ nhớ cùng một lúc.
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,     # shuffle=True: xáo trộn thứ tự ảnh mỗi epoch
    num_workers=2, pin_memory=torch.cuda.is_available(),     # num_workers: số luồng đọc dữ liệu song song
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,        # shuffle=False: giữ nguyên thứ tự khi đánh giá
    num_workers=2, pin_memory=torch.cuda.is_available(),        # pin_memory: tăng tốc chuyển dữ liệu sang GPU
)

print("-" * 60)                                                                  # in dòng kẻ phân cách cho dễ đọc log
print("Các lớp phân loại (classes):", class_names)                                # in ra danh sách 4 lớp đã đọc được
print(f"Tổng số ảnh: {len(full_train)} | Train: {len(train_dataset)} | Validation: {len(val_dataset)}")  # in số lượng ảnh từng tập
print("-" * 60)                                                                  # in dòng kẻ phân cách kết thúc
# Kết quả mong đợi: tổng 6776 ảnh, train khoảng 5421, validation khoảng
# 1355. Nếu số liệu in ra khác hẳn, có thể dataset add vào Kaggle bị
# thiếu hoặc sai, cần kiểm tra lại đường dẫn DATA_DIR.

# ==========================================================================
# CELL 4 - ĐỊNH NGHĨA KIẾN TRÚC MẠNG: ImprovedResSENet
# ==========================================================================
# Kiến trúc gồm 4 khối cải tiến (ImprovedBlock) xếp nối tiếp, tăng dần
# số kênh đặc trưng theo thứ tự 24 -> 48 -> 80 -> 112. Mỗi khối kết hợp
# đồng thời 3 kỹ thuật hiện đại: depthwise separable convolution,
# residual connection và SE-attention, đúng yêu cầu thể hiện cải tiến so
# với kiến trúc CNN cơ bản theo mục 6.3 của đề bài.

# ----- (1) Khối Squeeze-and-Excitation: cơ chế chú ý theo kênh -----
class SEBlock(nn.Module):
    """
    Sau khi trích xuất nhiều kênh đặc trưng, không phải kênh nào cũng
    quan trọng như nhau cho việc phân loại. Khối này học cách chấm điểm
    độ quan trọng của từng kênh (giá trị từ 0 đến 1, qua hàm Sigmoid),
    rồi nhân ngược lại vào feature map gốc - kênh quan trọng được giữ
    nguyên hoặc khuếch đại, kênh ít quan trọng bị giảm ảnh hưởng.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)                # tính số chiều rút gọn ở tầng giữa
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                 # nén mỗi kênh thành 1 giá trị duy nhất
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),           # giảm số chiều từ channels xuống reduced
            nn.ReLU(inplace=True),                               # thêm tính phi tuyến
            nn.Linear(reduced, channels, bias=False),             # khôi phục lại đúng số chiều ban đầu
            nn.Sigmoid(),                                          # ép giá trị về khoảng 0-1
        )

    def forward(self, x):
        b, c, _, _ = x.shape                  # lấy ra batch size và số kênh, bỏ qua chiều cao/rộng
        y = self.avg_pool(x).view(b, c)       # nén feature map, đổi shape về (batch, channels)
        y = self.fc(y).view(b, c, 1, 1)       # tính hệ số quan trọng, đổi lại shape để nhân được với x
        return x * y                            # nhân ngược hệ số quan trọng vào feature map gốc


# ----- (2) Khối Depthwise Separable Convolution: giảm số tham số -----
class DepthwiseSeparableConv(nn.Module):
    """
    Tách một phép tích chập thông thường thành 2 bước nhỏ hơn: depthwise
    (xử lý từng kênh riêng lẻ, không trộn kênh) và pointwise (dùng bộ
    lọc 1x1 để trộn thông tin giữa các kênh). Cách tách này giúp giảm
    đáng kể số lượng tham số cần huấn luyện so với tích chập thông
    thường, trong khi vẫn giữ được khả năng trích xuất đặc trưng.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            stride=stride, groups=in_channels, bias=False,      # groups=in_channels tạo ra depthwise convolution
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # trộn kênh bằng bộ lọc 1x1
        self.bn = nn.BatchNorm2d(out_channels)                   # chuẩn hóa theo lô sau tích chập
        self.relu = nn.ReLU(inplace=True)                        # hàm kích hoạt phi tuyến

    def forward(self, x):
        x = self.depthwise(x)     # bước 1: xử lý từng kênh riêng lẻ
        x = self.pointwise(x)      # bước 2: trộn thông tin giữa các kênh
        x = self.bn(x)               # chuẩn hóa kết quả
        return self.relu(x)            # áp dụng hàm kích hoạt trước khi trả ra


# ----- (3) Khối cải tiến hoàn chỉnh: kết hợp cả 3 kỹ thuật -----
class ImprovedBlock(nn.Module):
    """
    Khối xử lý chính, lặp lại 4 lần trong toàn mạng. Mỗi khối thực hiện
    lần lượt: tích chập tách biệt theo chiều sâu, cơ chế chú ý theo
    kênh, cộng kết nối tắt với đầu vào gốc, dropout chống overfitting,
    và gộp giảm kích thước feature map.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.15, downsample=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=1)  # nhánh xử lý chính
        self.se = SEBlock(out_channels)                                          # cơ chế chú ý theo kênh
        self.dropout = nn.Dropout2d(dropout_p)                                   # dropout cho dữ liệu dạng ảnh
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()             # giảm kích thước feature map đi một nửa

        # Khi số kênh đầu vào và đầu ra khác nhau, không thể cộng trực
        # tiếp 2 tensor có số kênh khác nhau. Phép projection dùng tích
        # chập 1x1 để đưa đầu vào gốc về đúng số kênh của đầu ra, trước
        # khi thực hiện phép cộng kết nối tắt.
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # đổi số kênh của nhánh tắt
                nn.BatchNorm2d(out_channels),                                       # chuẩn hóa sau khi đổi kênh
            )
        else:
            self.projection = nn.Identity()   # nếu số kênh không đổi, giữ nguyên đầu vào gốc

    def forward(self, x):
        identity = self.projection(x)    # giữ lại nhánh tắt, đã đồng bộ số kênh nếu cần
        out = self.conv(x)                 # bước 1: tích chập tách biệt theo chiều sâu
        out = self.se(out)                  # bước 2: cơ chế chú ý theo kênh
        out = out + identity                 # bước 3: cộng kết nối tắt (residual connection)
        out = self.dropout(out)               # bước 4: dropout chống overfitting
        out = self.pool(out)                   # bước 5: giảm kích thước feature map
        return out


# ----- (4) Kiến trúc toàn mạng: ImprovedResSENet -----
class ImprovedResSENet(nn.Module):
    """
    Kiến trúc hoàn chỉnh gồm một tầng xử lý ban đầu (stem), bốn khối cải
    tiến nối tiếp tăng dần số kênh từ 24 đến 112, và hai tầng kết nối
    đầy đủ ở cuối để đưa ra quyết định phân loại.

    Input:  (batch_size, 3, 256, 256) - ảnh màu RGB, kích thước 256x256
    Output: (batch_size, num_classes) - raw logits, chưa qua Softmax
    """
    def __init__(self, num_classes):
        super().__init__()

        # Stem: xử lý ban đầu khi ảnh vừa vào mạng, dùng tích chập thông
        # thường vì đây là tầng đầu tiên, xử lý ảnh thô chỉ có 3 kênh màu.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),  # tích chập đầu vào, 3 kênh sang 24 kênh
            nn.BatchNorm2d(24),                                       # chuẩn hóa theo lô
            nn.ReLU(inplace=True),                                     # hàm kích hoạt
        )

        # Bốn khối cải tiến, tăng dần số kênh theo thứ tự 24 -> 48 -> 80
        # -> 112, để mô hình có đủ dung lượng học các đặc trưng phức tạp
        # trong khi vẫn giữ tổng tham số dưới ngưỡng quy định của đề bài.
        self.block1 = ImprovedBlock(24, 48, dropout_p=0.10)
        self.block2 = ImprovedBlock(48, 80, dropout_p=0.10)
        self.block3 = ImprovedBlock(80, 112, dropout_p=0.15)
        self.block4 = ImprovedBlock(112, 112, dropout_p=0.15)

        # Phần đưa ra quyết định phân loại cuối cùng
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))   # nén feature map về 1 giá trị mỗi kênh
        self.flatten = nn.Flatten()                         # trải phẳng thành vector 1 chiều
        self.fc1 = nn.Linear(112, 96)                         # tầng kết nối đầy đủ thứ nhất
        self.relu = nn.ReLU(inplace=True)                       # hàm kích hoạt
        self.dropout = nn.Dropout(0.15)                           # dropout cho dữ liệu dạng vector
        self.fc2 = nn.Linear(96, num_classes)                       # tầng kết nối đầy đủ cuối cùng, ra đúng số lớp

    def forward(self, x):
        x = self.stem(x)              # xử lý ban đầu
        x = self.block1(x)              # khối cải tiến 1
        x = self.block2(x)                # khối cải tiến 2
        x = self.block3(x)                  # khối cải tiến 3
        x = self.block4(x)                    # khối cải tiến 4
        x = self.global_pool(x)                 # nén feature map về 1 giá trị mỗi kênh
        x = self.flatten(x)                       # trải phẳng thành vector
        x = self.relu(self.fc1(x))                  # tầng kết nối đầy đủ thứ nhất, qua ReLU
        x = self.dropout(x)                           # dropout trước tầng cuối
        x = self.fc2(x)                                 # tầng kết nối đầy đủ cuối, ra điểm số mỗi lớp
        return x   # trả về raw logits, không thêm Softmax theo đúng quy định mục 14.2.5 của đề bài


# Khởi tạo mô hình và kiểm tra tổng số tham số
model = ImprovedResSENet(num_classes=NUM_CLASSES).to(DEVICE)   # tạo model, chuyển sang đúng thiết bị (GPU/CPU)
params = count_parameters(model)                                  # đếm tổng số tham số có thể huấn luyện
print(f"Tổng số tham số của mô hình: {params:,}")                  # in ra số tham số để kiểm tra
assert params < 100000, "Mô hình vượt ngưỡng 100,000 tham số quy định!"  # kiểm tra điều kiện bắt buộc của đề bài
print("Kiểm tra tham số: hợp lệ.")                                    # xác nhận đã đạt điều kiện

# ==========================================================================
# CELL 5 - HUẤN LUYỆN MÔ HÌNH
# ==========================================================================
# Huấn luyện mạng nơ-ron qua 40 epoch. Mỗi epoch gồm 2 giai đoạn: train
# (học từ tập huấn luyện) và validation (đánh giá trên tập chưa từng học
# qua trong quá trình huấn luyện).

def train_one_epoch(model, loader, criterion, optimizer):
    """Chạy 1 epoch train: mô hình học từ dữ liệu và tự chỉnh trọng số."""
    model.train()                                       # chuyển model sang chế độ huấn luyện
    total_loss, correct, total = 0.0, 0, 0               # khởi tạo các biến cộng dồn theo epoch
    for images, labels in loader:                          # lặp qua từng batch ảnh và nhãn
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # chuyển dữ liệu sang đúng thiết bị
        optimizer.zero_grad()              # xóa gradient tích lũy từ lần lặp trước
        logits = model(images)               # đưa ảnh qua model, nhận về raw logits
        loss = criterion(logits, labels)      # tính sai số giữa dự đoán và nhãn thật
        loss.backward()                         # tính gradient cho toàn bộ trọng số
        optimizer.step()                          # cập nhật trọng số theo gradient vừa tính

        total_loss += loss.item() * images.size(0)        # cộng dồn loss theo số ảnh trong batch
        correct += (logits.argmax(dim=1) == labels).sum().item()  # cộng dồn số dự đoán đúng
        total += labels.size(0)                              # cộng dồn tổng số ảnh đã xử lý
    return total_loss / total, correct / total                # trả về loss và accuracy trung bình của epoch


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Chạy 1 epoch validation: chỉ đánh giá, không học hay chỉnh trọng số."""
    model.eval()                                         # chuyển model sang chế độ đánh giá
    total_loss, correct, total = 0.0, 0, 0                 # khởi tạo các biến cộng dồn
    for images, labels in loader:                            # lặp qua từng batch của tập validation
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # chuyển dữ liệu sang đúng thiết bị
        logits = model(images)                                  # đưa ảnh qua model
        loss = criterion(logits, labels)                          # tính sai số

        total_loss += loss.item() * images.size(0)                  # cộng dồn loss
        correct += (logits.argmax(dim=1) == labels).sum().item()      # cộng dồn số dự đoán đúng
        total += labels.size(0)                                         # cộng dồn tổng số ảnh
    return total_loss / total, correct / total                            # trả về loss và accuracy trung bình


# Cấu hình hàm loss và thuật toán tối ưu
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
# label_smoothing=0.05 giúp mô hình không quá tự tin vào một dự đoán
# duy nhất, hỗ trợ tổng quát hóa tốt hơn trên dữ liệu mới.

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# AdamW là thuật toán tối ưu quyết định chỉnh trọng số theo hướng nào và
# với mức độ bao nhiêu, dựa trên gradient tính được ở mỗi lần lặp.

# ReduceLROnPlateau tự động giảm learning rate xuống một nửa khi val_loss
# không giảm sau 4 epoch liên tiếp, giúp mô hình học chậm và kỹ hơn khi
# gần đạt điểm tối ưu, hạn chế dao động mạnh trong giai đoạn cuối.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

# Biến lưu lại lịch sử huấn luyện, dùng để vẽ đồ thị ở Cell 6
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc = 0.0          # lưu lại giá trị validation accuracy tốt nhất
best_state = None       # lưu lại bộ trọng số tại epoch tốt nhất, vì epoch
                          # cuối cùng không phải lúc nào cũng là epoch tốt nhất

print("\n=== TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH ===")
for epoch in range(1, EPOCHS + 1):                                      # lặp qua từng epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)  # huấn luyện 1 epoch
    val_loss, val_acc = evaluate(model, val_loader, criterion)            # đánh giá trên tập validation
    scheduler.step(val_loss)   # báo cho scheduler val_loss epoch này, để tự quyết định có giảm learning rate không
    current_lr = optimizer.param_groups[0]["lr"]                            # lấy learning rate hiện tại để in log

    history["train_loss"].append(train_loss)        # lưu lại train loss của epoch này
    history["val_loss"].append(val_loss)               # lưu lại validation loss
    history["train_acc"].append(train_acc)                # lưu lại train accuracy
    history["val_acc"].append(val_acc)                       # lưu lại validation accuracy

    if val_acc > best_acc:                                     # kiểm tra nếu epoch này tốt hơn các epoch trước
        best_acc = val_acc                                       # cập nhật lại giá trị tốt nhất
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # lưu lại toàn bộ trọng số

    print(f"Epoch {epoch:02d}/{EPOCHS} "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}")   # in log theo dõi từng epoch

# Nạp lại bộ trọng số tốt nhất (không phải bộ trọng số ở epoch cuối)
# trước khi đánh giá chi tiết và export ở các Cell tiếp theo.
if best_state is not None:
    model.load_state_dict(best_state)         # nạp lại đúng bộ trọng số tốt nhất vào model

print(f"\nBest validation accuracy: {best_acc:.4f}")   # in ra accuracy tốt nhất đạt được trong quá trình huấn luyện

# ==========================================================================
# CELL 6 - ĐÁNH GIÁ CHI TIẾT VÀ VẼ ĐỒ THỊ
# ==========================================================================
# Tính các chỉ số đánh giá Accuracy, Precision, Recall, F1-score cho
# từng lớp, vẽ confusion matrix và đồ thị loss/accuracy theo epoch, dùng
# để đưa vào báo cáo.

model.eval()                            # chuyển model sang chế độ đánh giá
all_preds, all_labels = [], []          # khởi tạo 2 danh sách lưu kết quả dự đoán và nhãn thật
with torch.no_grad():                     # tắt tính gradient vì chỉ cần dự đoán, không cần học
    for images, labels in val_loader:       # lặp qua từng batch của tập validation
        images = images.to(DEVICE)            # chuyển ảnh sang đúng thiết bị
        outputs = model(images)                 # đưa ảnh qua model, nhận về raw logits
        _, predicted = torch.max(outputs, 1)      # lấy ra lớp có điểm số cao nhất làm dự đoán
        all_preds.extend(predicted.cpu().numpy())   # lưu lại kết quả dự đoán của batch này
        all_labels.extend(labels.numpy())             # lưu lại nhãn thật của batch này

all_preds = np.array(all_preds)            # chuyển danh sách dự đoán sang dạng mảng NumPy
all_labels = np.array(all_labels)          # chuyển danh sách nhãn thật sang dạng mảng NumPy
# Truyền labels=range(num_classes) rõ ràng để tránh lỗi nếu một lớp nào
# đó không xuất hiện trong tập validation.
all_class_indices = list(range(len(class_names)))   # tạo danh sách chỉ số của 4 lớp

report_text = classification_report(
    all_labels, all_preds, labels=all_class_indices,
    target_names=class_names, digits=4, zero_division=0,
)   # tính precision, recall, f1-score cho từng lớp
macro_f1 = f1_score(all_labels, all_preds, labels=all_class_indices, average="macro", zero_division=0)  # tính F1 trung bình các lớp
overall_acc = (all_preds == all_labels).sum() / len(all_labels)   # tính accuracy tổng quan

print("\n[BÁO CÁO HIỆU NĂNG CHI TIẾT - ImprovedResSENet]")   # in tiêu đề báo cáo
print(report_text)                                              # in bảng chỉ số chi tiết theo từng lớp
print(f"Overall Accuracy: {overall_acc*100:.2f}%")                # in accuracy tổng quan
print(f"Macro F1-score: {macro_f1:.4f}")                             # in F1-score trung bình

# Vẽ confusion matrix
cm = confusion_matrix(all_labels, all_preds)                          # tính ma trận nhầm lẫn
plt.figure(figsize=(8, 6))                                              # tạo khung hình mới
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)  # vẽ heatmap
plt.title("Confusion Matrix - ImprovedResSENet")                          # đặt tiêu đề hình
plt.ylabel("Nhãn thực tế (Actual)")                                         # đặt tên trục y
plt.xlabel("Nhãn dự đoán (Predicted)")                                        # đặt tên trục x
plt.tight_layout()                                                             # tự căn lề cho hình không bị cắt chữ
plt.savefig("confusion_matrix_ImprovedResSENet.png", dpi=200)                    # lưu hình ra file ảnh
plt.close()                                                                        # đóng hình hiện tại để vẽ hình mới

# Vẽ đồ thị loss theo epoch
plt.figure(figsize=(8, 5))                                                          # tạo khung hình mới
plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train Loss")             # vẽ đường train loss
plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Validation Loss")             # vẽ đường validation loss
plt.xlabel("Epoch")                                                                         # đặt tên trục x
plt.ylabel("Loss")                                                                            # đặt tên trục y
plt.title("Training & Validation Loss - ImprovedResSENet")                                      # đặt tiêu đề hình
plt.legend()                                                                                       # hiện chú thích đường
plt.grid(alpha=0.3)                                                                                   # thêm lưới mờ cho dễ đọc
plt.tight_layout()                                                                                       # tự căn lề
plt.savefig("loss_curve_ImprovedResSENet.png", dpi=200)                                                    # lưu hình ra file
plt.close()                                                                                                   # đóng hình hiện tại

# Vẽ đồ thị accuracy theo epoch
plt.figure(figsize=(8, 5))                                                              # tạo khung hình mới
plt.plot(range(1, EPOCHS + 1), [a * 100 for a in history["train_acc"]], label="Train Accuracy")  # vẽ đường train accuracy
plt.plot(range(1, EPOCHS + 1), [a * 100 for a in history["val_acc"]], label="Validation Accuracy")  # vẽ đường validation accuracy
plt.xlabel("Epoch")                                                                                     # đặt tên trục x
plt.ylabel("Accuracy (%)")                                                                                 # đặt tên trục y
plt.title("Training & Validation Accuracy - ImprovedResSENet")                                              # đặt tiêu đề hình
plt.legend()                                                                                                   # hiện chú thích đường
plt.grid(alpha=0.3)                                                                                               # thêm lưới mờ
plt.tight_layout()                                                                                                   # tự căn lề
plt.savefig("accuracy_curve_ImprovedResSENet.png", dpi=200)                                                            # lưu hình ra file
plt.close()                                                                                                                # đóng hình hiện tại

print("\nĐã lưu 3 file ảnh: confusion_matrix_ImprovedResSENet.png, "
      "loss_curve_ImprovedResSENet.png, accuracy_curve_ImprovedResSENet.png")   # thông báo đã lưu xong các file ảnh

# CELL 7 — XUẤT FILE MÔ HÌNH .pt ĐỂ NỘP BÀI (KHÔNG ĐƯỢC SỬA CELL NÀY)
# ==========================================================================
# ###########################################
# DO NOT MODIFY THIS SECTION
# ###########################################
# Cấu trúc câu lệnh trong khối này PHẢI giữ nguyên vẹn theo đúng quy định
# bắt buộc mục 14.2.3 và 14.4 của đề bài - đây là khối lệnh chuẩn hóa,
# giúp giảng viên load tự động hàng loạt file mô hình của các nhóm vào
# hệ thống chấm điểm tự động.

model.eval()
example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)

model_name = f"{GroupID}_DeepLearningProject_TrainedModel.pt"
traced_model.save(model_name)
print("Model saved:", os.path.abspath(model_name))