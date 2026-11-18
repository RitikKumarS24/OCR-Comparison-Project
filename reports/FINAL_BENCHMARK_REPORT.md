# Comparative Evaluation of Classical and Deep Learning OCR Engines Across Multilingual and Degraded Document Conditions

## Abstract
Optical Character Recognition (OCR) systems face substantial performance variance across script families, document degradation modes, and hardware constraints. This study presents an end-to-end empirical benchmark comparing five representative OCR architectures—**Tesseract (LSTM)**, **EasyOCR (CRAFT + CRNN)**, **PaddleOCR (PP-OCRv5/v6)**, **docTR (FAST + CRNN-VGG16)**, and **SuryaOCR (Vision-Language Transformer)**—across **100 bilingual test images** spanning **English (Latin)** and **Hindi (Devanagari)** under five controlled image conditions (`original`, `noisy`, `blurry`, `rotated`, and `low_resolution`). Using a standardized OpenCV preprocessing pipeline (`raw` vs. `preprocessed`, totaling **1,000 OCR evaluations**) and Unicode NFC-normalized error metrics (CER and WER), we demonstrate that **SuryaOCR** and **PaddleOCR** tie for peak English character accuracy (**99.79%**, CER **0.0021**), while **SuryaOCR** leads Hindi recognition (**97.71%** raw / **97.15%** preprocessed). Furthermore, domain-specific OpenCV preprocessing reduces Character Error Rate by **69.3% on EasyOCR (English)** and **81.9% on Tesseract (English)**, establishing clear Pareto trade-offs between real-time latency (<0.35s) and transformer-grade multilingual accuracy.

---

## 1. System Architecture & Methodology

### 1.1 Evaluated OCR Engines
| Engine | Version / Backend | Detection Architecture | Recognition Architecture | Native Multilingual Support |
| :--- | :--- | :--- | :--- | :--- |
| **Tesseract OCR** | v5.x (`pytesseract`) | Connected-Component / Line Finding | Bi-Directional LSTM + CTC | Yes (`eng`, `hin` tessdata) |
| **EasyOCR** | v1.7+ (PyTorch CUDA) | CRAFT (Character Region Awareness) | ResNet/VGG + BiLSTM + CTC | Yes (`en`, `hi` / Devanagari) |
| **PaddleOCR** | v3.7 (`PaddlePaddle`) | PP-OCRv5/v6 DBNet | SVTR / MobileNet + CTC | Yes (`en`, `hi` pipelines) |
| **docTR** | v1.1.0 (PyTorch CUDA) | `fast_base` (Reparameterized) | `crnn_vgg16_bn` | Latin-only vocabulary |
| **SuryaOCR** | v0.17.1 (PyTorch CUDA) | Vision Transformer Layout/Det | Encoder-Decoder Transformer | Yes (90+ languages incl. Hindi) |

### 1.2 Bilingual Dataset & Controlled Distortion Matrix
* **Base Corpus:** 20 curated ground-truth text documents (10 English, 10 Hindi) stored in `dataset/ground_truth/` and indexed in `dataset/metadata.csv`.
* **Five Experimental Conditions (100 Total Images):**
  1. `original` (20 images): Clean high-contrast synthetic document renders.
  2. `noisy` (20 images): Additive Gaussian and salt-and-pepper scanner noise.
  3. `blurry` (20 images): Out-of-focus Gaussian/motion blur kernels.
  4. `rotated` (20 images): Skewed document orientations simulating misaligned scans.
  5. `low_resolution` (20 images): Downsampled pixel grids simulating low-DPI captures.

### 1.3 Adaptive OpenCV Preprocessing Pipeline (`preprocessing/image_preprocessor.py`)
Each image was evaluated both **raw** and after passing through an adaptive OpenCV restoration pipeline tailored to its degradation mode:
* **Standard Mode (`original`, `rotated`, `blurry`):** Grayscale conversion, projection-profile/contour deskewing, CLAHE contrast enhancement, and unsharp masking.
* **Binary Mode (`noisy`):** Median + Gaussian denoising, Otsu adaptive binarization, and morphological noise cleanup.
* **Low-Resolution Mode (`low_resolution`):** Bicubic $2\times$ super-resolution upscaling followed by CLAHE contrast sharpening.

### 1.4 Evaluation Metrics (`evaluation/metrics.py`)
All ground-truth and hypothesis strings undergo canonical **Unicode NFC normalization** (`unicodedata.normalize("NFC", text)`) to prevent false mismatches on composite Devanagari vowel signs (*matras*) and *nukta* characters:
* **Character Error Rate (CER):** $\text{CER} = \frac{S_c + D_c + I_c}{N_c}$
* **Word Error Rate (WER):** $\text{WER} = \frac{S_w + D_w + I_w}{N_w}$
* **Character & Word Accuracy (%):** $\max(0, (1 - \text{Error Rate}) \times 100)$
* **Runtime Efficiency:** Per-image wall-clock inference latency (`time.perf_counter()`) after GPU warm-up, plus process RSS memory delta (`psutil`).

---

## 2. Quantitative Experimental Results (1,000 Total Runs)

### 2.1 Overall Performance Summary by Engine, Pipeline & Language
| Engine | Pipeline | Language | Avg CER ↓ | Avg WER ↓ | Char Acc (%) ↑ | Word Acc (%) ↑ | Avg Latency (s) ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PaddleOCR** | Preprocessed | English | **0.0021** | **0.0138** | **99.79%** | **98.62%** | 5.8397 |
| **SuryaOCR** | Preprocessed | English | **0.0021** | 0.0214 | **99.79%** | 97.86% | 2.6908 |
| **SuryaOCR** | Raw | English | 0.0023 | 0.0279 | 99.77% | 97.21% | 2.7596 |
| **PaddleOCR** | Raw | English | 0.0023 | 0.0216 | 99.77% | 97.84% | 4.9242 |
| **Tesseract** | Preprocessed | English | 0.0078 | 0.0697 | 99.22% | 93.03% | 0.2683 |
| **Tesseract** | Raw | English | 0.0432 | 0.1831 | 95.68% | 82.05% | 0.2720 |
| **docTR** | Raw | English | 0.0545 | 0.1727 | 94.55% | 83.09% | 0.2067 |
| **docTR** | Preprocessed | English | 0.0588 | 0.1946 | 94.12% | 80.54% | **0.2037** |
| **EasyOCR** | Preprocessed | English | 0.0607 | 0.2888 | 93.93% | 71.12% | 0.3419 |
| **EasyOCR** | Raw | English | 0.1976 | 0.4062 | 80.24% | 59.92% | **0.1589** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SuryaOCR** | Raw | Hindi | **0.0229** | **0.0877** | **97.71%** | **91.23%** | 2.6551 |
| **SuryaOCR** | Preprocessed | Hindi | 0.0285 | 0.0991 | 97.15% | 90.09% | 2.6442 |
| **Tesseract** | Preprocessed | Hindi | 0.1157 | 0.3481 | 88.43% | 65.19% | 0.2040 |
| **PaddleOCR** | Preprocessed | Hindi | 0.1212 | 0.2858 | 87.88% | 71.42% | 9.5764 |
| **EasyOCR** | Preprocessed | Hindi | 0.1563 | 0.5429 | 84.37% | 45.71% | 0.2195 |
| **Tesseract** | Raw | Hindi | 0.1570 | 0.4034 | 84.30% | 59.66% | 0.2050 |
| **PaddleOCR** | Raw | Hindi | 0.1600 | 0.3275 | 84.00% | 67.25% | 8.1805 |
| **EasyOCR** | Raw | Hindi | 0.2644 | 0.5881 | 73.56% | 41.47% | **0.1985** |
| **docTR** | Preprocessed | Hindi | 0.8631 | 0.9901 | 13.69% | 2.31% | 0.2092 |
| **docTR** | Raw | Hindi | 0.8663 | 0.9921 | 13.37% | 2.64% | 0.2113 |

### 2.2 Visual Analytics
* **Accuracy by Engine & Language:** `../results/graphs/01_accuracy_by_engine_language.png`
* **Raw vs. Preprocessed CER Reduction:** `../results/graphs/02_raw_vs_preprocessed_impact.png`
* **Distortion Robustness Heatmap:** `../results/graphs/03_robustness_by_condition_heatmap.png`
* **Speed vs. Accuracy Pareto Frontier:** `../results/graphs/04_speed_vs_accuracy_pareto.png`
* **CER vs. WER Comparison:** `../results/graphs/05_cer_wer_comparison.png`

---

## 3. Key Technical Findings & Failure-Mode Analysis

1. **Transformer Superiority on Complex Indic Scripts (Hindi):**
   Devanagari poses unique segmentation challenges due to the continuous horizontal headline (*shirorekha*) connecting characters and stacked consonant conjuncts. **SuryaOCR** achieves **97.71% character accuracy** on Hindi because its vision-language encoder-decoder processes whole line context rather than relying on fragile character-level CTC segmentation.
2. **Impact of OpenCV Preprocessing:**
   * Classical and CRNN-based engines benefit dramatically from explicit image restoration: **Tesseract** English CER dropped from `0.0432` to `0.0078` (**81.9% error reduction**), and **EasyOCR** English CER dropped from `0.1976` to `0.0607` (**69.3% error reduction**).
   * Conversely, **SuryaOCR** slightly preferred raw grayscale/RGB textures on Hindi (`97.71%` raw vs. `97.15%` preprocessed), as aggressive binary thresholding can occasionally clip thin Devanagari *matra* strokes.
3. **Out-of-Vocabulary Script Behavior (`docTR` on Hindi):**
   Because pretrained `docTR` checkpoints (`crnn_vgg16_bn`) use a Latin-only character set, running `docTR` on Hindi yields `86.31%` CER (mapping Devanagari glyphs to visually similar Latin letters), underscoring the necessity of script-aware model selection in multilingual pipelines.
4. **Deployment Recommendations (Speed vs. Accuracy Trade-Off):**
   * **High-Throughput Real-Time English Pipelines (<0.30s/image):** **Tesseract (with OpenCV preprocessing)** delivers `99.22%` character accuracy at `0.268s` per image with minimal RAM overhead (`0.90 MB`).
   * **Mission-Critical Multilingual & Archival Digitization:** **SuryaOCR** provides the strongest cross-lingual accuracy (`99.79%` English, `97.71%` Hindi) at `~2.65s` per image on an RTX 3050 GPU.