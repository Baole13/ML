# Chạy benchmark local

Chạy local được; Colab không bắt buộc. Full benchmark tải hai dataset Kaggle
và chạy PCA/HOG cùng các mô hình CNN/DINOv2. GPU NVIDIA được khuyến nghị.
Trên CPU notebook vẫn chạy nhưng thời gian sẽ dài hơn đáng kể.

## 1. Tạo môi trường Python

Khuyến nghị Python 3.11 để tương thích tốt với PyTorch và các thư viện notebook.
Từ thư mục gốc repo, trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r notebook/requirements.txt
```

Nếu máy có NVIDIA GPU và muốn dùng CUDA, cài `torch`/`torchvision` theo lệnh
phù hợp với Windows và phiên bản CUDA do trang cài đặt chính thức của PyTorch
tạo ra; sau đó cài các dependency còn lại ở trên. Xác nhận CUDA với
`python -c "import torch; print(torch.cuda.is_available())"`.

Trên Linux/macOS, tạo và kích hoạt virtual environment theo hệ điều hành rồi
cài cùng file `notebook/requirements.txt`.

## 2. Cấu hình Kaggle

`kagglehub` tải hai dataset mà notebook đang dùng. Ngoài Kaggle Notebook, xác
thực một lần bằng lệnh sau; KaggleHub sẽ hỏi token theo cách ẩn và lưu thông tin
xác thực cục bộ:

```powershell
python -c "import kagglehub; kagglehub.login()"
```

Tạo API token trong trang [Kaggle API settings](https://www.kaggle.com/settings/api).
Không dán token vào notebook, repo hoặc file `.env` đã theo dõi bởi Git. Tài liệu
chính thức: [KaggleHub authentication](https://github.com/Kaggle/kagglehub#authenticate).

## 3. Khởi chạy notebook

```powershell
jupyter lab
```

Mở `notebook/Animal_Visual_Search_Clean_Benchmark_v1.ipynb`, chọn kernel của
`.venv-ml`, đổi `QUICK_MODE = False`, restart kernel và chạy tất cả cell từ đầu.
Notebook tự đặt output vào `results/official_full/` khi chạy trong repo local.
Trên Colab, output tiếp tục nằm ở `/content/animal_visual_search_clean_v1/`.

Trước full run, có thể chạy một lần với `QUICK_MODE = True` để xác nhận tải dữ
liệu, FAISS và mô hình hoạt động. Kết quả Quick Mode chỉ để smoke-test, không
được dùng thay benchmark chính thức.

## 4. Lưu kết quả và artifact

Giữ trong Git các kết quả nhẹ cần cho báo cáo: `benchmark_results.csv`,
`integrity_summary.csv`, `fixed_clean_split.csv`, `resnet_test_predictions.csv`
và cấu hình lần chạy. Không commit checkpoint `.pt`, embedding `.npy`, index
FAISS hoặc gallery ảnh. Phần xuất serving artifact chỉ thực hiện sau khi full
benchmark hoàn tất, từ chính gallery/classifier của lần chạy đó.
