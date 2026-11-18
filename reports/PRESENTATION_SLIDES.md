# Slide Deck: Comparative Evaluation of Classical & Deep Learning OCR Engines

---

## Slide 1: Title Slide
* **Title:** Comparative Evaluation of Classical and Deep Learning OCR Engines Across Multilingual and Degraded Document Conditions
* **Scope:** 5 OCR Architectures × 2 Languages (English & Hindi) × 5 Document Conditions × 2 Pipelines = **1,000 Benchmark Runs**

---

## Slide 2: Problem Statement & Motivation
* Real-world scanned documents suffer from **noise, blur, skew/rotation, and low resolution**.
* Multilingual Indian document digitization requires simultaneous accuracy on **Latin (English)** and **Devanagari (Hindi)**—where continuous *shirorekha* headlines and stacked *matras* challenge traditional character segmentation.
* **Research Question:** How do classical LSTM engines compare against modern CNN-RNN and Vision Transformer OCR models in accuracy, preprocessing sensitivity, and latency?

---

## Slide 3: Evaluated OCR Architectures
1. **Tesseract v5:** Classical connected-component detection + BiLSTM/CTC.
2. **EasyOCR:** CRAFT character-region detector + ResNet-BiLSTM-CTC.
3. **PaddleOCR v3.7:** PP-OCRv5/v6 DBNet detector + SVTR/MobileNet recognizer.
4. **docTR v1.1:** Reparameterized FAST detector + CRNN-VGG16 (Latin-only baseline).
5. **SuryaOCR v0.17:** End-to-end Vision-Language Encoder-Decoder Transformer.

---

## Slide 4: Dataset & Adaptive OpenCV Preprocessing Pipeline
* **100 Test Images:** 20 ground-truth passages (10 English, 10 Hindi) across `original`, `noisy`, `blurry`, `rotated`, and `low_resolution`.
* **Adaptive OpenCV Restoration:**
  * *Standard (`original`, `rotated`, `blurry`):* Grayscale → Deskew → CLAHE → Unsharp Mask.
  * *Binary (`noisy`):* Median/Gaussian Denoise → Otsu Thresholding → Morphological Cleanup.
  * *Low-Res (`low_resolution`):* $2\times$ Bicubic Upscaling → CLAHE Sharpening.

---

## Slide 5: Evaluation Methodology (Fair & Script-Accurate)
* **Unicode NFC Normalization:** Essential for Devanagari vowel signs (*matras*) and *nukta* characters so identical visual text is never penalized.
* **Metrics:** Character Error Rate (**CER**), Word Error Rate (**WER**), Character/Word Accuracy (%), Wall-Clock Latency (after CUDA warm-up), and RAM delta.

---

## Slide 6: English Recognition Results
* *(Show `results/graphs/01_accuracy_by_engine_language.png` & `02_raw_vs_preprocessed_impact.png`)*
* **PaddleOCR** & **SuryaOCR** tie at **99.79% Character Accuracy** (`0.0021` CER).
* OpenCV preprocessing cuts **Tesseract** error by **81.9%** (`95.68%` → `99.22%`) and **EasyOCR** error by **69.3%** (`80.24%` → `93.93%`).

---

## Slide 7: Hindi (Devanagari) Recognition Results
* **SuryaOCR** leads decisively at **97.71% Character Accuracy** (`0.0229` CER), outperforming **Tesseract** (`88.43%`) and **PaddleOCR** (`87.88%`) by ~9.3 percentage points.
* **docTR** scores `13.69%` due to its Latin-only output vocabulary—illustrating script-mismatch failure modes.

---

## Slide 8: Robustness Across Distortion Conditions
* *(Show `results/graphs/03_robustness_by_condition_heatmap.png`)*
* Vision Transformers (**SuryaOCR**) and DBNet pipelines (**PaddleOCR**) maintain >97% accuracy even under heavy blur, rotation, and low resolution.

---

## Slide 9: Speed vs. Accuracy Pareto Trade-Off
* *(Show `results/graphs/04_speed_vs_accuracy_pareto.png`)*
* **Real-Time Tier (<0.30s/img):** Preprocessed **Tesseract** (`99.22%` English accuracy in `0.268s`).
* **High-Accuracy Multilingual Tier (~2.65s/img):** **SuryaOCR** (`99.79%` English, `97.71%` Hindi).

---

## Slide 10: Conclusion & Future Work
* **Key Takeaway:** No single engine dominates every constraint—**preprocessed Tesseract** wins real-time English throughput, while **SuryaOCR** wins bilingual and Devanagari accuracy.
* **Future Work:** Hybrid router selecting Tesseract for clean English and SuryaOCR for Hindi/degraded pages; extending benchmark to handwritten Devanagari and multi-column tables.