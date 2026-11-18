import os
import time
import logging
import psutil
import cv2

# Disable MKLDNN/oneDNN PIR conversion bug on PaddlePaddle 3.3.x Windows CPU
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR


class PaddleOCREngine:
    """Wrapper for PaddleOCR v3.7 supporting English and Hindi (Devanagari)."""

    def __init__(self):
        self.name = "PaddleOCR"
        # Initialize English and Hindi pipelines once at startup
        self.ocr_en = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.ocr_hi = PaddleOCR(
            lang="hi",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image_input, lang: str = "English") -> dict:
        """
        Run PaddleOCR on either a file path (str) or a BGR NumPy image array.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        else:
            img = image_input

        engine = self.ocr_hi if lang.lower() == "hindi" else self.ocr_en

        start_time = time.perf_counter()
        raw_results = engine.predict(img)
        latency = time.perf_counter() - start_time

        # Extract recognized text strings from PaddleOCR v3 result objects
        extracted_lines = []
        for res in raw_results:
            if isinstance(res, dict) and "rec_texts" in res:
                extracted_lines.extend(res["rec_texts"])
            elif hasattr(res, "get") and res.get("rec_texts"):
                extracted_lines.extend(res.get("rec_texts"))

        mem_after = process.memory_info().rss / (1024 * 1024)
        cleaned_text = " ".join(" ".join(extracted_lines).split())

        return {
            "engine": self.name,
            "text": cleaned_text,
            "latency_sec": round(latency, 4),
            "ram_delta_mb": round(max(0.0, mem_after - mem_before), 2),
        }


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Initializing PaddleOCR (loading cached PP-OCR English & Hindi models)...")
    engine = PaddleOCREngine()

    eng_sample = os.path.join(project_root, "dataset", "original", "eng_01.png")
    hin_sample = os.path.join(project_root, "dataset", "original", "hin_01.png")

    print("--- Testing PaddleOCR ---")
    print("English Result:", engine.predict(eng_sample, lang="English"))
    print("Hindi Result:  ", engine.predict(hin_sample, lang="Hindi"))