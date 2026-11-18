# Comparative OCR Benchmark: Classical vs. Deep Learning Engines (English & Hindi)

An end-to-end, reproducible benchmarking framework evaluating five OCR engines (**Tesseract**, **EasyOCR**, **PaddleOCR**, **docTR**, and **SuryaOCR**) across **100 bilingual document images** (`English` and `Hindi`) under **5 controlled image degradation conditions** (`original`, `noisy`, `blurry`, `rotated`, and `low_resolution`) and **2 pipelines** (`raw` vs. `preprocessed` via OpenCV)—totaling **1,000 quantitative OCR runs**.

---

## Key Empirical Highlights

| Rank | Engine | English Char Acc (Preprocessed) | Hindi Char Acc (Best) | Avg Latency / Image | Key Takeaway |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **SuryaOCR** (`v0.17.1`) | **99.79%** (`0.0021` CER) | **97.71%** (`0.0229` CER) | ~2.65 s | Best overall multilingual & Devanagari accuracy |
| **2** | **PaddleOCR** (`v3.7`) | **99.79%** (`0.0021` CER) | 87.88% (`0.1212` CER) | 5.84–9.58 s | Tied #1 on English; strong layout robustness |
| **3** | **Tesseract** (`v5.x`) | **99.22%** (`0.0078` CER) | 88.43% (`0.1157` CER) | **0.20–0.27 s** | Best real-time speed/accuracy trade-off with OpenCV |
| **4** | **EasyOCR** (`v1.7+`) | 93.93% (`0.0607` CER) | 84.37% (`0.1563` CER) | **0.16–0.34 s** | +13.7% English gain from OpenCV preprocessing |
| **5** | **docTR** (`v1.1.0`) | 94.55% (`0.0545` CER) | 13.69% (`0.8631` CER) | **0.20–0.21 s** | Fast Latin-only baseline; out-of-vocab on Devanagari |

---

## Project Structure

```text
OCR_Comparison_Project/
├── dataset/
│   ├── original/               # 20 clean baseline images (10 English, 10 Hindi)
│   ├── noisy/                  # 20 images with Gaussian & salt-and-pepper noise
│   ├── blurry/                 # 20 images with Gaussian/motion blur
│   ├── rotated/                # 20 skewed/rotated document images
│   ├── low_resolution/         # 20 downsampled low-DPI images
│   ├── ground_truth/           # UTF-8 reference ground-truth text files
│   ├── metadata.csv            # Dataset index & image dimensions
│   └── prepare_dataset.py      # Reproducible dataset & distortion generator
├── preprocessing/
│   └── image_preprocessor.py   # Adaptive OpenCV restoration (deskew, CLAHE, Otsu, 2x upscale)
├── ocr_engines/
│   ├── tesseract_ocr.py        # Classical LSTM + CTC wrapper
│   ├── easyocr_engine.py       # CRAFT + BiLSTM wrapper
│   ├── paddleocr_engine.py     # PP-OCRv5/v6 wrapper
│   ├── doctr_engine.py         # Mindee docTR (FAST + CRNN-VGG16) wrapper
│   └── surya_engine.py         # Surya Vision-Language Transformer wrapper
├── evaluation/
│   ├── metrics.py              # Unicode NFC normalization, CER & WER calculation
│   ├── run_benchmark.py        # 1,000-run automated benchmark suite with checkpoints
│   └── generate_visuals.py     # 300-DPI publication chart generator
├── results/
│   ├── raw_outputs/
│   │   └── benchmark_results.csv
│   ├── summary_by_engine.csv
│   ├── summary_by_condition.csv
│   └── graphs/                 # 5 high-resolution comparative charts
├── reports/
│   ├── FINAL_BENCHMARK_REPORT.md
│   └── PRESENTATION_SLIDES.md
└── requirements.txt