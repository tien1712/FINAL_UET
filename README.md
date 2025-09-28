# Travel Mode Prediction using LLM with FAISS Retrieval

Dự án dự đoán phương thức di chuyển sử dụng Large Language Model (LLM) kết hợp với FAISS retrieval để tìm kiếm các chuyến đi tương tự.

## Tổng quan

Dự án này sử dụng:
- **LLM**: Google Gemini để dự đoán phương thức di chuyển
- **FAISS**: Để tìm kiếm các chuyến đi tương tự từ dữ liệu lịch sử
- **Retrieval-Augmented Generation (RAG)**: Kết hợp thông tin từ các chuyến đi tương tự để cải thiện độ chính xác

## Cấu trúc dự án

```
├── baseline/           # Code baseline với LLM
│   ├── final.py       # Script chính để chạy prediction
│   ├── prompt.py      # Template prompt cho LLM
│   └── f1-score.py    # Script tính F1-score
├── FAISS/             # Code FAISS retrieval
│   ├── main.py        # Script tạo FAISS index
│   ├── retrieval.py   # Hàm retrieval cho PSRC dataset
│   └── retrieval_swissmetro.py  # Hàm retrieval cho Swissmetro dataset
├── data/              # Dữ liệu
│   ├── PSRC_Seatle/   # Dataset PSRC Seattle
│   ├── Swissmetro/    # Dataset Swissmetro
│   └── Optima/        # Dataset Optima
├── db/                # FAISS indexes và embeddings
└── results/           # Kết quả prediction
```

## Cài đặt

1. **Clone repository:**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

3. **Cấu hình API keys:**
Tạo file `.env` và thêm các Google API keys:
```
GOOGLE_API_KEY_1=your_api_key_1
GOOGLE_API_KEY_2=your_api_key_2
...
GOOGLE_API_KEY_11=your_api_key_11
```

## Sử dụng

### 1. Tạo FAISS index
```bash
cd FAISS
python main.py
```

### 2. Chạy prediction
```bash
cd baseline
python final.py
```

### 3. Tính F1-score
```bash
python f1-score.py
```

## Datasets

- **PSRC Seattle**: Household Travel Survey data
- **Swissmetro**: Swiss transportation survey
- **Optima**: Transportation choice data

## Phương thức di chuyển được dự đoán

- **Drive**: Đi bằng ô tô cá nhân
- **Walk**: Đi bộ
- **Transit**: Sử dụng phương tiện công cộng
- **Bike/Micromobility**: Đi xe đạp hoặc phương tiện micromobility

## Kết quả

Dự án đạt được độ chính xác cao trong việc dự đoán phương thức di chuyển bằng cách:
- Sử dụng thông tin từ các chuyến đi tương tự
- Kết hợp với context cá nhân của người dùng
- Áp dụng retrieval-augmented generation

## Tác giả

Nguyễn Hữu Tiến - Đồ án tốt nghiệp