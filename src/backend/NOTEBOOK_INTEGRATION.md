# Nối notebook Colab → Backend

`docs/plan.md` yêu cầu thêm **một** cell `Serving Export` vào notebook để xuất
gói artifact hoàn chỉnh, rồi backend load gói đó. Đây là các bước cụ thể.

## 1. Thêm cell cuối cùng vào notebook

Chép `src/backend/scripts/serving_export.py` lên Colab (upload file hoặc để
trong working dir), rồi thêm **một** code cell ở cuối notebook:

```python
# ---- Serving Export (cell cuối cùng của notebook) ----
import sys
sys.path.append("/content")          # thư mục chứa serving_export.py
from serving_export import export_serving_package

export_serving_package(
    output_dir=OUTPUT_DIR / "artifacts",   # OUTPUT_DIR đã có sẵn trong notebook
    gallery_dino=gallery_dino,             # (N, 384) train embeddings
    gallery_labels_dino=gallery_labels_dino,
    gallery_paths_dino=gallery_paths_dino, # list[str] đường dẫn ảnh train
    dino_clf=dino_clf,                      # LinearSVC đã fit trong notebook
    label_names=label_names,                # sorted(species)
    model_name="dinov2_vits14",
    version="clean-v1",                     # đổi khi export bản mới
    image_size=IMG_SIZE_CNN,                # 224
)
```

Cell này sẽ:
- L2-normalize embeddings (khớp notebook).
- Bọc `LinearSVC` trong `CalibratedClassifierCV` → classifier có
  `predict_proba` (confidence thật). Tương thích sklearn cũ và mới.
- Build FAISS `IndexFlatIP`.
- Copy ảnh gallery vào `artifacts/gallery/<species>/<id>.jpg` với **ID ổn định**.
- Ghi `manifest.json` (preprocessing contract) + `class_mapping.json`.

## 2. Tải gói artifact về máy / server

Trên Colab:

```python
!zip -r artifacts.zip OUTPUT_DIR/artifacts   # hoặc đường dẫn tuyệt đối
from google.colab import files
files.download("artifacts.zip")
```

Giải nén vào `src/backend/artifacts/` (hoặc set `ARTIFACTS_DIR`).

## 3. Chạy backend

```bash
cd src/backend
pip install -r requirements.txt
ENCODER=dinov2 ARTIFACTS_DIR=./artifacts uvicorn app.main:app --reload
```

> Lần đầu `ENCODER=dinov2` sẽ tải weights DINOv2 ViT-S/14 từ torch.hub. Muốn
> smoke-test không cần weights, dùng bộ synthetic:
>
> ```bash
> python scripts/make_synthetic_artifacts.py ./artifacts
> ENCODER=mock ARTIFACTS_DIR=./artifacts uvicorn app.main:app --reload
> ```

## 4. Kiểm tra nhanh

```bash
curl -F "image=@some_tiger.jpg" -F "k=5" http://127.0.0.1:8000/api/v1/search
curl http://127.0.0.1:8000/model-info
```

## Lưu ý quan trọng về contract

Backend **không** hard-code preprocessing. Nó đọc `image_size`, `normalization`,
`mean`, `std`, `embedding_dim`, `metric`, `index_type` từ `manifest.json`. Nếu
bạn đổi transform trong notebook, hãy export lại manifest — backend sẽ tự áp
dụng đúng transform, không cần sửa code.
