import os
import time
import psutil
import cv2
import torch
import easyocr


class EasyOCREngine:
    """Wrapper for EasyOCR v1.7 (CRAFT Detector + CRNN Recognizer) with GPU support."""

    def __init__(self):
        self.name = "EasyOCR"
        self.use_gpu = torch.cuda.is_available()
        # Initialize readers once at startup so model loading is excluded from inference latency
        self.reader_en = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
        self.reader_hi = easyocr.Reader(["hi", "en"], gpu=self.use_gpu, verbose=False)

    def predict(self, image_input, lang: str = "English") -> dict:
        """
        Run EasyOCR on either a file path (str) or a BGR NumPy image array.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        else:
            img = image_input

        reader = self.reader_hi if lang.lower() == "hindi" else self.reader_en

        start_time = time.perf_counter()
        # detail=0 returns a list of recognized text strings in reading order
        results = reader.readtext(img, detail=0, paragraph=True)
        latency = time.perf_counter() - start_time

        mem_after = process.memory_info().rss / (1024 * 1024)
        cleaned_text = " ".join(" ".join(results).split())

        return {
            "engine": self.name,
            "text": cleaned_text,
            "latency_sec": round(latency, 4),
            "ram_delta_mb": round(max(0.0, mem_after - mem_before), 2),
        }


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Initializing EasyOCR (downloading CRAFT & CRNN weights if first run)...")
    engine = EasyOCREngine()

    eng_sample = os.path.join(project_root, "dataset", "original", "eng_01.png")
    hin_sample = os.path.join(project_root, "dataset", "original", "hin_01.png")

    print("--- Testing EasyOCR ---")
    print("English Result:", engine.predict(eng_sample, lang="English"))
    print("Hindi Result:  ", engine.predict(hin_sample, lang="Hindi"))