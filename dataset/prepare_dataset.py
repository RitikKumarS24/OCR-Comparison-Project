import os
import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# 1. Define Project Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = {
    "original": os.path.join(BASE_DIR, "original"),
    "ground_truth": os.path.join(BASE_DIR, "ground_truth"),
    "blurry": os.path.join(BASE_DIR, "blurry"),
    "noisy": os.path.join(BASE_DIR, "noisy"),
    "rotated": os.path.join(BASE_DIR, "rotated"),
    "low_resolution": os.path.join(BASE_DIR, "low_resolution"),
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# 2. Load Windows Fonts (English + Devanagari/Hindi)
# ---------------------------------------------------------
def load_font(font_candidates, size):
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

WIN_FONTS = r"C:\Windows\Fonts"
FONT_SANS = load_font([os.path.join(WIN_FONTS, "arial.ttf"), os.path.join(WIN_FONTS, "calibri.ttf")], 34)
FONT_SERIF = load_font([os.path.join(WIN_FONTS, "times.ttf"), os.path.join(WIN_FONTS, "georgia.ttf")], 34)
FONT_MONO = load_font([os.path.join(WIN_FONTS, "consola.ttf"), os.path.join(WIN_FONTS, "cour.ttf")], 32)
FONT_BOLD = load_font([os.path.join(WIN_FONTS, "arialbd.ttf")], 40)
# Nirmala.ttc is Windows' built-in Devanagari/Hindi font
FONT_HINDI = load_font([os.path.join(WIN_FONTS, "Nirmala.ttc"), os.path.join(WIN_FONTS, "mangal.ttf")], 36)

# ---------------------------------------------------------
# 3. Curated Diverse Dataset Samples (14 English + 6 Hindi)
# ---------------------------------------------------------
SAMPLES = [
    # --- English Samples (Academic, Invoices, Technical, Scene/Signage, Dense Text) ---
    ("eng_01", "English", "Academic Paragraph", FONT_SERIF, (255, 255, 255), (15, 15, 15),
     "Optical Character Recognition converts scanned documents into searchable digital text."),
    ("eng_02", "English", "Academic Paragraph", FONT_SANS, (255, 255, 255), (20, 20, 20),
     "Deep learning models combine convolutional feature extraction with sequence decoding."),
    ("eng_03", "English", "Invoice & Finance", FONT_MONO, (255, 255, 255), (10, 10, 10),
     "INVOICE #INV-2026-0849 | Date: 18-11-2026 | Total Amount Due: $1,485.50 USD"),
    ("eng_04", "English", "Invoice & Finance", FONT_MONO, (250, 250, 245), (25, 25, 25),
     "Item: NVIDIA RTX 3050 GPU | Qty: 02 | Unit Price: $249.99 | Tax (18%): $89.99"),
    ("eng_05", "English", "Technical & Code", FONT_MONO, (245, 248, 255), (15, 25, 60),
     "CER = (Substitutions + Deletions + Insertions) / Total_Reference_Characters"),
    ("eng_06", "English", "Technical & Code", FONT_MONO, (255, 255, 255), (0, 0, 0),
     "python main.py --engine paddleocr --lang en --input dataset/original/eng_01.png"),
    ("eng_07", "English", "Signage & Wayfinding", FONT_BOLD, (240, 245, 255), (10, 40, 110),
     "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING - BLOCK B, FLOOR 3"),
    ("eng_08", "English", "Signage & Wayfinding", FONT_BOLD, (255, 250, 230), (140, 20, 20),
     "CAUTION: HIGH VOLTAGE LABORATORY EQUIPMENT - AUTHORIZED PERSONNEL ONLY"),
    ("eng_09", "English", "Medical & Prescription", FONT_SANS, (255, 255, 255), (20, 20, 20),
     "Rx: Paracetamol 500mg Tablet - Take 1 tablet twice daily after meals for 5 days."),
    ("eng_10", "English", "Legal & Administrative", FONT_SERIF, (255, 255, 255), (15, 15, 15),
     "Certified True Copy: University Examination Roll No. 2026CS1042 Verified."),
    ("eng_11", "English", "Mixed Alphanumeric", FONT_SANS, (255, 255, 255), (15, 15, 15),
     "Flight AI-302 | Gate B14 | Seat 18A | Departure 06:45 IST | PNR: X9K7M2"),
    ("eng_12", "English", "Multi-Line Paragraph", FONT_SANS, (255, 255, 255), (20, 20, 20),
     "Image preprocessing techniques such as Gaussian filtering and Otsu thresholding\n"
     "improve foreground-background contrast before character segmentation."),
    ("eng_13", "English", "Multi-Line Paragraph", FONT_SERIF, (255, 255, 255), (15, 15, 15),
     "Vision Transformers divide input document images into fixed-size patches\n"
     "and capture long-range spatial dependencies across complex page layouts."),
    ("eng_14", "English", "Punctuation & Symbols", FONT_MONO, (255, 255, 255), (10, 10, 10),
     "Email: research.lab@university.edu.in | Tel: +91-120-2456789 | Ref: [A-99/B]"),

    # --- Hindi Samples (Devanagari Script: Academic, Administrative, Public Notice, Mixed) ---
    ("hin_01", "Hindi", "Academic Sentence", FONT_HINDI, (255, 255, 255), (15, 15, 15),
     "ऑप्टिकल कैरेक्टर रिकग्निशन तकनीक मुद्रित दस्तावेजों को डिजिटल पाठ में बदलती है।"),
    ("hin_02", "Hindi", "University Notice", FONT_HINDI, (255, 255, 255), (15, 15, 15),
     "कंप्यूटर विज्ञान एवं इंजीनियरिंग विभाग की प्रायोगिक परीक्षा सोमवार को आयोजित होगी।"),
    ("hin_03", "Hindi", "Public Signage", FONT_HINDI, (245, 250, 255), (10, 45, 105),
     "केंद्रीय पुस्तकालय एवं अनुसंधान केंद्र - प्रवेश द्वार संख्या दो"),
    ("hin_04", "Hindi", "Administrative Text", FONT_HINDI, (255, 255, 255), (20, 20, 20),
     "भारत सरकार द्वारा जारी डिजिटल इंडिया कार्यक्रम का मुख्य उद्देश्य तकनीकी सशक्तिकरण है।"),
    ("hin_05", "Hindi", "Multi-Line Hindi", FONT_HINDI, (255, 255, 255), (15, 15, 15),
     "छवि प्रसंस्करण में शोर हटाने और कंट्रास्ट सुधारने से\n"
     "अक्षर पहचान की सटीकता में उल्लेखनीय वृद्धि होती है।"),
    ("hin_06", "Hindi", "Hindi with Numerals", FONT_HINDI, (255, 255, 255), (15, 15, 15),
     "परीक्षा केंद्र संख्या 104 में कुल 250 विद्यार्थियों की उपस्थिति दर्ज की गई।"),
]

# ---------------------------------------------------------
# 4. Helper Functions: Render Clean Image & Apply Degradations
# ---------------------------------------------------------
def render_clean_image(text, font, bg_color, text_color):
    lines = text.split("\n")
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)

    line_heights = []
    max_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_width = max(max_width, w)
        line_heights.append(h)

    pad_x, pad_y, line_spacing = 60, 50, 18
    img_w = max_width + 2 * pad_x
    img_h = sum(line_heights) + line_spacing * (len(lines) - 1) + 2 * pad_y

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = pad_y
    for line, h in zip(lines, line_heights):
        draw.text((pad_x, y), line, font=font, fill=text_color)
        y += h + line_spacing

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def create_degraded_versions(bgr_img):
    h, w = bgr_img.shape[:2]

    # 1. Blurry: 7x7 Gaussian blur
    blurry = cv2.GaussianBlur(bgr_img, (7, 7), 2.2)

    # 2. Noisy: Deterministic Gaussian noise (seed=42 for reproducibility)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 28, bgr_img.shape).astype(np.float32)
    noisy = np.clip(bgr_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. Rotated: +10 degree rotation around center with white background
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, 10.0, 1.0)
    cos_val = np.abs(rot_mat[0, 0])
    sin_val = np.abs(rot_mat[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        bgr_img, rot_mat, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )

    # 4. Low Resolution: Downscale to 35% and upscale back with nearest-neighbor pixelation
    small_w = max(1, int(w * 0.35))
    small_h = max(1, int(h * 0.35))
    downsampled = cv2.resize(bgr_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    low_res = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_NEAREST)

    return blurry, noisy, rotated, low_res


# ---------------------------------------------------------
# 5. Generate All Images, Ground-Truth Files & Metadata CSV
# ---------------------------------------------------------
def main():
    metadata_path = os.path.join(BASE_DIR, "metadata.csv")
    rows = []

    for sample_id, lang, category, font, bg_color, text_color, text in SAMPLES:
        clean_img = render_clean_image(text, font, bg_color, text_color)
        blurry_img, noisy_img, rotated_img, low_res_img = create_degraded_versions(clean_img)

        img_filename = f"{sample_id}.png"
        gt_filename = f"{sample_id}.txt"

        cv2.imwrite(os.path.join(DIRS["original"], img_filename), clean_img)
        cv2.imwrite(os.path.join(DIRS["blurry"], img_filename), blurry_img)
        cv2.imwrite(os.path.join(DIRS["noisy"], img_filename), noisy_img)
        cv2.imwrite(os.path.join(DIRS["rotated"], img_filename), rotated_img)
        cv2.imwrite(os.path.join(DIRS["low_resolution"], img_filename), low_res_img)

        with open(os.path.join(DIRS["ground_truth"], gt_filename), "w", encoding="utf-8") as f:
            f.write(text)

        rows.append({
            "sample_id": sample_id,
            "image_file": img_filename,
            "ground_truth_file": gt_filename,
            "language": lang,
            "category": category,
            "char_count": len(text),
            "word_count": len(text.split()),
            "width_px": clean_img.shape[1],
            "height_px": clean_img.shape[0],
        })

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"SUCCESS: Generated {len(SAMPLES)} clean images, {len(SAMPLES)} ground-truth files,")
    print(f"         and {len(SAMPLES) * 4} degraded images across 4 distortion categories.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()