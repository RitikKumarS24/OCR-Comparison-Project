import os
import time
import psutil
import cv2
import numpy as np
import pytesseract


class TesseractEngine:
    """Wrapper for Tesseract v5.4 LSTM OCR supporting English and Hindi."""

    def __init__(self):
        self.name = "Tesseract"
        tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_exe):
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe

        # Set TESSDATA_PREFIX via environment variable (avoids Windows quote parsing bugs)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tessdata_dir = os.path.join(project_root, "tessdata")
        os.environ["TESSDATA_PREFIX"] = self.tessdata_dir

    def predict(self, image_input, lang: str = "English") -> dict:
        """
        Run Tesseract OCR on either a file path (str) or a BGR NumPy image array.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        else:
            img = image_input

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tess_lang = "hin" if lang.lower() == "hindi" else "eng"
        custom_config = "--oem 3 --psm 6"

        start_time = time.perf_counter()
        raw_text = pytesseract.image_to_string(rgb, lang=tess_lang, config=custom_config)
        latency = time.perf_counter() - start_time

        mem_after = process.memory_info().rss / (1024 * 1024)
        cleaned_text = " ".join(raw_text.split())

        return {
            "engine": self.name,
            "text": cleaned_text,
            "latency_sec": round(latency, 4),
            "ram_delta_mb": round(max(0.0, mem_after - mem_before), 2),
        }


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine = TesseractEngine()

    eng_sample = os.path.join(project_root, "dataset", "original", "eng_01.png")
    hin_sample = os.path.join(project_root, "dataset", "original", "hin_01.png")

    print("--- Testing Tesseract OCR ---")
    print("English Result:", engine.predict(eng_sample, lang="English"))
    print("Hindi Result:  ", engine.predict(hin_sample, lang="Hindi"))