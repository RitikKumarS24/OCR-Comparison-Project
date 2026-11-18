import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "results", "graphs")
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "reports", "OCR_Benchmark_Presentation.pptx")

# Color Palette (Academic Navy & Slate)
NAVY = RGBColor(24, 48, 89)
TEAL = RGBColor(27, 158, 119)
DARK_TEXT = RGBColor(33, 37, 41)
MUTED_TEXT = RGBColor(90, 98, 106)
LIGHT_BG = RGBColor(244, 246, 249)
WHITE = RGBColor(255, 255, 255)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, subtitle_text=""):
    # Top accent bar
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.6))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.88), Inches(12.1), Inches(0.35))
        stf = sub_box.text_frame
        sp = stf.paragraphs[0]
        sp.text = subtitle_text
        sp.font.size = Pt(14)
        sp.font.color.rgb = MUTED_TEXT


def add_bullets(slide, bullets, left=0.6, top=1.45, width=12.1, height=5.5, font_size=17):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = f"• {lead}: "
            r1.font.bold = True
            r1.font.size = Pt(font_size)
            r1.font.color.rgb = NAVY

            r2 = p.add_run()
            r2.text = rest
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = DARK_TEXT
        else:
            r = p.add_run()
            r.text = f"• {item}"
            r.font.size = Pt(font_size)
            r.font.color.rgb = DARK_TEXT


def add_chart_image(slide, filename, left, top, width=None, height=None):
    img_path = os.path.join(GRAPHS_DIR, filename)
    if os.path.exists(img_path):
        kwargs = {}
        if width is not None:
            kwargs["width"] = Inches(width)
        if height is not None:
            kwargs["height"] = Inches(height)
        slide.shapes.add_picture(img_path, Inches(left), Inches(top), **kwargs)


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, NAVY)

    accent = s1.shapes.add_shape(1, Inches(0.8), Inches(1.8), Inches(1.4), Inches(0.08))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(2.2))
    ttf = t_box.text_frame
    ttf.word_wrap = True
    tp = ttf.paragraphs[0]
    tp.text = "Comparative Evaluation of Classical and Deep Learning OCR Engines"
    tp.font.size = Pt(38)
    tp.font.bold = True
    tp.font.color.rgb = WHITE

    sub_p = ttf.add_paragraph()
    sub_p.space_before = Pt(14)
    sub_p.text = "Bilingual (English & Hindi) Benchmarking Across Degraded Document Conditions"
    sub_p.font.size = Pt(20)
    sub_p.font.color.rgb = RGBColor(200, 218, 240)

    meta_box = s1.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.2))
    mtf = meta_box.text_frame
    mp = mtf.paragraphs[0]
    mp.text = "5 OCR Engines  |  100 Bilingual Images  |  5 Distortion Conditions  |  1,000 Automated Benchmark Runs"
    mp.font.size = Pt(15)
    mp.font.bold = True
    mp.font.color.rgb = TEAL

    # -------------------------------------------------------------------------
    # Slide 2: Problem Statement & Motivation
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2, WHITE)
    add_header(s2, "Problem Statement & Motivation", "Why Multilingual & Degraded OCR Evaluation Matters")
    add_bullets(s2, [
        ("Real-World Scan Degradations", "Digitized archives and mobile captures suffer from sensor noise, motion blur, skew/rotation, and low DPI resolution."),
        ("Script Complexity (Latin vs. Devanagari)", "Hindi features a continuous headline (shirorekha) connecting glyphs plus stacked vowel matras that break traditional character segmentation."),
        ("Preprocessing Sensitivity", "Classical LSTM engines depend heavily on binarization and deskewing, whereas modern Vision Transformers learn spatial invariance."),
        ("Research Objective", "Quantify Character Error Rate (CER), Word Error Rate (WER), and inference latency across 5 representative OCR architectures."),
    ])

    # -------------------------------------------------------------------------
    # Slide 3: Evaluated OCR Architectures
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3, WHITE)
    add_header(s3, "Evaluated OCR Architectures", "From Classical Connected-Component LSTM to Vision-Language Transformers")

    rows, cols = 6, 4
    table_shape = s3.shapes.add_table(rows, cols, Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.2))
    table = table_shape.table
    headers = ["Engine", "Paradigm", "Detection / Recognition Backbone", "Multilingual Scope"]
    data = [
        ["Tesseract v5", "Classical + LSTM", "Connected Components + BiLSTM / CTC", "English + Hindi (tessdata)"],
        ["EasyOCR v1.7", "Deep CNN-RNN", "CRAFT Detector + ResNet-BiLSTM-CTC", "English + Hindi (Devanagari)"],
        ["PaddleOCR v3.7", "Industrial Pipeline", "PP-OCRv5/v6 DBNet + SVTR / MobileNet", "English + Hindi Pipelines"],
        ["docTR v1.1", "Document CNN-RNN", "Reparameterized FAST + CRNN-VGG16", "Latin-Only Vocabulary"],
        ["SuryaOCR v0.17", "Vision Transformer", "ViT Layout/Det + Encoder-Decoder Transformer", "90+ Languages (incl. Hindi)"],
    ]

    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE

    for r_idx, row in enumerate(data, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13.5)
            p.font.bold = (c_idx == 0)
            p.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------------------
    # Slide 4: Dataset & Adaptive OpenCV Preprocessing
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4, WHITE)
    add_header(s4, "Dataset & Adaptive OpenCV Pipeline", "100 Bilingual Images Evaluated Both Raw and Preprocessed (1,000 Total Runs)")
    add_bullets(s4, [
        ("Balanced Bilingual Corpus", "20 ground-truth passages (10 English, 10 Hindi) rendered across 5 conditions = 100 test images."),
        ("Standard Mode (original, rotated, blurry)", "Grayscale conversion -> contour/projection deskewing -> CLAHE contrast enhancement -> unsharp masking."),
        ("Binary Mode (noisy)", "Median + Gaussian denoising -> Otsu adaptive binarization -> morphological cleanup."),
        ("Low-Resolution Mode (low_resolution)", "2x bicubic super-resolution upscaling followed by CLAHE edge sharpening."),
        ("Unicode NFC Evaluation", "Canonical Unicode NFC normalization prevents false CER penalties on composite Devanagari matras."),
    ])

    # -------------------------------------------------------------------------
    # Slide 5: Accuracy by Engine & Language (Chart 1)
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5, WHITE)
    add_header(s5, "Overall Accuracy by Engine & Language", "SuryaOCR & PaddleOCR Tie on English (99.79%); SuryaOCR Dominates Hindi (97.71%)")
    add_chart_image(s5, "01_accuracy_by_engine_language.png", left=0.6, top=1.45, width=7.8)
    add_bullets(s5, [
        ("English Leaders", "PaddleOCR and SuryaOCR both reach 99.79% character accuracy (0.0021 CER)."),
        ("Hindi Champion", "SuryaOCR achieves 97.71% raw / 97.15% preprocessed—beating Tesseract & PaddleOCR by ~9%."),
        ("Out-of-Vocab Failure", "docTR scores 13.69% on Hindi due to its Latin-only character set."),
    ], left=8.6, top=1.6, width=4.2, font_size=15)

    # -------------------------------------------------------------------------
    # Slide 6: Impact of OpenCV Preprocessing (Chart 2)
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6, WHITE)
    add_header(s6, "Impact of OpenCV Preprocessing on CER", "Classical & CRNN Engines Gain Massively from Image Restoration")
    add_chart_image(s6, "02_raw_vs_preprocessed_impact.png", left=0.5, top=1.55, width=12.3)

    # -------------------------------------------------------------------------
    # Slide 7: Robustness Across Distortion Conditions (Chart 3)
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7, WHITE)
    add_header(s7, "Distortion Robustness Heatmap", "Performance Breakdown Across Original, Noisy, Blurry, Rotated & Low-Res")
    add_chart_image(s7, "03_robustness_by_condition_heatmap.png", left=0.5, top=1.6, width=12.3)

    # -------------------------------------------------------------------------
    # Slide 8: Speed vs. Accuracy Pareto Frontier (Chart 4)
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8, WHITE)
    add_header(s8, "Speed vs. Accuracy Pareto Trade-Off", "Selecting the Optimal Engine for Real-Time vs. Archival Workloads")
    add_chart_image(s8, "04_speed_vs_accuracy_pareto.png", left=0.6, top=1.45, width=7.7)
    add_bullets(s8, [
        ("Real-Time English (<0.30s)", "Preprocessed Tesseract delivers 99.22% accuracy in 0.268s with <1 MB RAM delta."),
        ("Multilingual Precision (~2.65s)", "SuryaOCR provides unmatched cross-lingual accuracy on RTX 3050 GPU."),
        ("CPU Bottleneck", "PaddleOCR takes 5.8–9.6s/image on CPU despite top English accuracy."),
    ], left=8.5, top=1.6, width=4.3, font_size=15)

    # -------------------------------------------------------------------------
    # Slide 9: CER vs. WER Error Analysis (Chart 5)
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9, WHITE)
    add_header(s9, "Character Error Rate (CER) vs. Word Error Rate (WER)", "Why Dual-Granularity Metrics Reveal True Document Usability")
    add_chart_image(s9, "05_cer_wer_comparison.png", left=0.6, top=1.45, width=7.7)
    add_bullets(s9, [
        ("Error Amplification", "A single misread character in a 10-letter word causes 100% WER but only 10% CER."),
        ("Search & Indexing Readiness", "PaddleOCR (0.0138 WER) and SuryaOCR (0.0214 WER) preserve >97.8% whole words intact."),
    ], left=8.5, top=1.7, width=4.3, font_size=15)

    # -------------------------------------------------------------------------
    # Slide 10: Conclusion & Engineering Recommendations
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10, WHITE)
    add_header(s10, "Conclusion & Engineering Recommendations", "Actionable Architecture Selection Guidelines")
    add_bullets(s10, [
        ("Best Overall Multilingual Engine", "Deploy SuryaOCR for bilingual English/Hindi archives and degraded historical scans (99.79% EN / 97.71% HI)."),
        ("Best Low-Latency English Pipeline", "Deploy Tesseract v5 paired with adaptive OpenCV preprocessing (99.22% accuracy at 0.27s/image)."),
        ("Preprocessing Rule of Thumb", "Always apply deskew + Otsu/CLAHE before classical/CRNN models, but pass raw RGB images to Vision Transformers."),
        ("Future Work", "Build a confidence-based hybrid router and expand evaluation to handwritten Devanagari and multi-column tables."),
    ])

    prs.save(OUTPUT_PPTX)
    print(f"Presentation saved successfully -> {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_presentation()