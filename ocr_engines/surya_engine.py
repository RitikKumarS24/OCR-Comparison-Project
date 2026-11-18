import os
import time
import psutil
import cv2
import torch
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


class SuryaEngine:
    """Wrapper for Surya OCR v0.17 (Pure-PyTorch Transformer OCR on CUDA/CPU)."""

    def __init__(self):
        self.name = "SuryaOCR"
        self.foundation_predictor = FoundationPredictor()
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)

    def predict(self, image_input, lang: str = "English") -> dict:
        """
        Run Surya OCR on either a file path (str) or a BGR NumPy image array.
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
        pil_img = Image.fromarray(rgb)

        start_time = time.perf_counter()
        predictions = self.rec_predictor([pil_img], det_predictor=self.det_predictor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time

        extracted_lines = []
        if predictions and hasattr(predictions[0], "text_lines"):
            extracted_lines = [
                line.text.strip()
                for line in predictions[0].text_lines
                if getattr(line, "text", None)
            ]

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
    print("Initializing Surya OCR (downloading PyTorch checkpoints on first run)...")
    engine = SuryaEngine()

    eng_sample = os.path.join(project_root, "dataset", "original", "eng_01.png")
    hin_sample = os.path.join(project_root, "dataset", "original", "hin_01.png")

    print("--- Testing Surya OCR ---")
    print("English Result:", engine.predict(eng_sample, lang="English"))
    print("Hindi Result:  ", engine.predict(hin_sample, lang="Hindi"))