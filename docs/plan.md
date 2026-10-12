**Đủ để chuyển sang làm backend ngay**, nhưng chưa nên coi model/artifact hiện tại là “production frozen”. Phần nghiên cứu đã đủ trưởng thành để backend hóa; thứ còn thiếu chủ yếu là **đóng gói model để serving**, không phải tiếp tục nghiên cứu thêm model.

Điểm mấu chốt: backend nên được thiết kế **model-agnostic** ngay từ đầu, để sau này thay DINOv2/ResNet/SigLIP2 mà API không phải sửa.

Hiện notebook đã có nền tảng tốt: clean fixed split, benchmark nhiều representation, ResNet checkpoint, DINOv2 gallery embeddings, labels, cross-dataset audit và FAISS-style retrieval. Tuy nhiên phần export hiện mới lưu `resnet18_clean_v1.pt`, `dino_gallery_embeddings.npy` và `dino_gallery_labels.npy`; **chưa lưu DINO LinearSVC classifier, gallery paths/metadata hay FAISS index**. Ngoài ra các path hiện trỏ vào Kaggle cache/Colab, backend deploy sang máy khác sẽ không dùng trực tiếp được.

Có một vấn đề quan trọng hơn nữa: **DINOv2 + LinearSVC hiện không cho probability/confidence chuẩn**. `LinearSVC` chỉ có `decision_function()`, không có `predict_proba()`. Vì sản phẩm của bạn muốn trả:

```text
Species: Tiger
Confidence: 92%
Top-K related images
```

thì trước bản backend cuối cùng chúng ta cần calibrate classifier hoặc đổi classification head. Mình nghiêng về:

```text
DINOv2 frozen embedding
        │
        ├── Logistic Regression / calibrated Linear SVM
        │        ↓
        │    species + probability
        │
        └── FAISS
                 ↓
             Top-K images
```

Như vậy cùng một representation phục vụ cả classification và retrieval, kiến trúc đẹp hơn việc ResNet đoán class còn DINO retrieve ảnh.

### Backend mình đề xuất

```text
Client
  │
  │ POST /api/v1/search
  │ image + k
  ▼
FastAPI
  │
  ▼
Image validation
  │
  ▼
Preprocessing
224 × 224
  │
  ▼
DINOv2 encoder
  │
  └──────────── embedding 384-D ──────────────┐
                    │                          │
                    ▼                          ▼
          Classification head             FAISS index
                    │                          │
             species + confidence          Top-K IDs
                    │                          │
                    └──────────┬───────────────┘
                               ▼
                         Metadata store
                               │
                               ▼
                         JSON response
```

API chính chỉ cần dạng:

```http
POST /api/v1/search
Content-Type: multipart/form-data

image=<file>
k=5
```

Response:

```json
{
  "prediction": {
    "species": "tiger",
    "confidence": 0.927
  },
  "results": [
    {
      "id": "tiger_0012",
      "species": "tiger",
      "similarity": 0.941,
      "image_url": "/gallery/tiger_0012.jpg"
    },
    {
      "id": "tiger_0048",
      "species": "tiger",
      "similarity": 0.916,
      "image_url": "/gallery/tiger_0048.jpg"
    }
  ],
  "model": {
    "encoder": "dinov2_vits14",
    "version": "v1"
  }
}
```

Ngoài ra chỉ cần `/health` và `/model-info` là đủ cho MVP.

### Trước khi viết backend, mình chỉ muốn thêm một cell `Serving Export`

Nó cần xuất đủ bộ:

```text
artifacts/
├── manifest.json
├── class_mapping.json
├── dino_classifier.joblib
├── dino_gallery_embeddings.npy
├── dino_gallery_labels.npy
├── dino_gallery_metadata.csv
├── dino_gallery.faiss
└── gallery/
    ├── ...
```

`manifest.json` nên cố định preprocessing:

```json
{
  "model": "dinov2_vits14",
  "image_size": 224,
  "normalization": "imagenet",
  "embedding_dim": 384,
  "metric": "cosine",
  "index_type": "IndexFlatIP",
  "version": "clean-v1"
}
```

Điều này cực kỳ quan trọng. Backend không được copy preprocessing bằng trí nhớ; nó phải đọc đúng contract mà notebook đã train/evaluate.

### Gallery cũng cần sửa cách lưu

Hiện notebook có:

```python
gallery_paths_dino
```

nhưng chưa export nó, và ngay cả khi export thì path sẽ kiểu:

```text
/root/.cache/kagglehub/datasets/...
```

Deploy backend sang máy khác là chết.

Nên chuyển gallery thành ID ổn định:

```text
animal90:tiger:0012
animal90:wolf:0038
```

và metadata:

| id | species | filename |
|---|---|---|
| tiger_0012 | tiger | gallery/tiger/tiger_0012.jpg |
| wolf_0038 | wolf | gallery/wolf/wolf_0038.jpg |

Backend chỉ cần map:

```text
FAISS position
→ image ID
→ metadata
→ image URL
```

không phụ thuộc filesystem Colab.

### Có cần chờ full 90-class benchmark không?

**Không cần chờ để bắt đầu backend.**

Nên làm song song:

```text
ML side
Full clean benchmark
calibration
OOD
SigLIP2
        │
        │ artifacts
        ▼
Backend interface
        ▲
        │
API / inference engine / gallery service
đã làm từ bây giờ
```

Miễn là backend sử dụng interface kiểu:

```python
class AnimalSearchEngine:
    def predict(...)
    def search(...)
```

thì sau này DINOv2 bị SigLIP2 vượt qua, chỉ đổi implementation bên trong. Frontend/API không thay đổi.

### Cái gì chưa cần làm trước backend?

Chưa cần:

- SigLIP2
- Grad-CAM
- ResNet scratch/frozen ablation
- OOD cheetah
- toàn bộ error taxonomy

Đó là **research/report enhancement**, không phải backend blocker.

Có hai thứ mình sẽ làm **trước hoặc trong lúc bắt đầu backend**:

1. **Serving artifact export** như trên.
2. **Calibrated confidence** vì đây là một phần output trực tiếp mà người dùng nhìn thấy.

OOD có thể thêm sau dưới dạng:

```json
"is_unknown": null
```

rồi sau này đổi thành `true/false` mà không phá API.

## Chốt

Hiện project đã ở trạng thái:

```text
Research prototype      ✅
Clean benchmark design  ✅
Retrieval engine        ✅
Model selection path    ✅
Backend architecture    ✅ ready to start

Serving artifacts       ⚠️ cần đóng gói
Confidence calibration  ⚠️ cần làm
Production deployment   ❌ chưa
```

**Mình sẽ chuyển sang backend ngay từ đây.** Nhưng trước khi viết FastAPI, bước đầu tiên mình muốn làm là nâng notebook thêm đúng một Phase: **`Serving Export`**, để xuất một package DINOv2 + classifier + FAISS + gallery metadata hoàn chỉnh. Sau đó backend sẽ load package đó trực tiếp, thay vì copy code từ Colab sang.