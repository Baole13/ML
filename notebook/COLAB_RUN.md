# Chạy benchmark trên Google Colab

## 1. Mở notebook

Upload duy nhất file `Animal_Visual_Search_Clean_Benchmark_v1.ipynb` lên Google Colab. Không cần upload dataset, virtual environment hoặc cache; KaggleHub tải dữ liệu tự động.

## 2. Thêm Kaggle token an toàn

1. Trong Colab, mở mục **Secrets** (biểu tượng chìa khóa ở thanh bên trái).
2. Tạo secret tên chính xác `KAGGLE_API_TOKEN` và dán giá trị token vào đó.
3. Bật quyền truy cập secret cho notebook hiện tại.
4. Không ghi token vào code cell hoặc notebook đã lưu.

KaggleHub tự đọc secret `KAGGLE_API_TOKEN` trong Colab. Xem [hướng dẫn xác thực chính thức](https://github.com/Kaggle/kagglehub#authenticate).

## 3. Chạy full benchmark

1. Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU khả dụng).
2. Chạy cell cài thư viện.
3. Trong cell cấu hình, đổi `QUICK_MODE = True` thành `QUICK_MODE = False`.
4. Chọn **Runtime → Restart session**, rồi **Run all**.
5. Đợi tới cell cuối báo hoàn thành. Notebook tải Animal-90 và Wild Animals rồi chạy audit và toàn bộ mô hình; full mode sẽ lâu hơn Quick Mode.

Kết quả được ghi dưới `/content/animal_visual_search_clean_v1/`. Tải về các file sau để đưa vào repo/báo cáo:

- `benchmark_results.csv`
- `integrity_summary.csv`
- `fixed_clean_split.csv`
- `resnet_test_predictions.csv`
- `run_config.json`
- `figures/`

Có thể đóng gói chúng từ một cell cuối:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

root = Path("/content/animal_visual_search_clean_v1")
keep = [
    "benchmark_results.csv",
    "integrity_summary.csv",
    "fixed_clean_split.csv",
    "resnet_test_predictions.csv",
    "run_config.json",
]
with ZipFile("benchmark_results.zip", "w", ZIP_DEFLATED) as zf:
    for name in keep:
        zf.write(root / name, name)
    for p in (root / "figures").glob("*"):
        if p.is_file():
            zf.write(p, f"figures/{p.name}")
files.download("benchmark_results.zip")
```

Chỉ báo cáo kết quả từ lần chạy `QUICK_MODE=False`; Quick Mode chỉ dùng để xác nhận pipeline.
