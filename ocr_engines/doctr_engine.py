import os
import time
import psutil
import cv2
import torch
from doctr.models import ocr_predictor


class DocTREngine:
    """Wrapper for Mindee docTR v1.1.0 (PyTorch backend with GPU support)."""

    def __init__(self):
        self.name = "docTR"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize two-stage predictor once at startup
        self.predictor = ocr_predictor(
            det_arch="fast_base",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=False,
        ).to(self.device)

    def predict(self, image_input, lang: str = "English") -> dict:
        """
        Run docTR on either a file path (str) or a BGR NumPy image array.
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        else:
            img = image_input

        # docTR expects a list of RGB NumPy arrays [H, W, 3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.perf_counter()
        result = self.predictor([rgb])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time

        mem_after = process.memory_info().rss / (1024 * 1024)
        cleaned_text = " ".join(result.render().split())

        return {
            "engine": self.name,
            "text": cleaned_text,
            "latency_sec": round(latency, 4),
            "ram_delta_mb": round(max(0.0, mem_after - mem_before), 2),
        }


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Initializing docTR (downloading fast_base & crnn_vgg16_bn weights if first run)...")
    engine = DocTREngine()

    eng_sample = os.path.join(project_root, "dataset", "original", "eng_01.png")
    hin_sample = os.path.join(project_root, "dataset", "original", "hin_01.png")

    print("--- Testing docTR ---")
    print("English Result:", engine.predict(eng_sample, lang="English"))
    print("Hindi Result:  ", engine.predict(hin_sample, lang="Hindi"))