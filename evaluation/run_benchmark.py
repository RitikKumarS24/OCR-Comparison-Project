import os
import sys
import gc
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_ocr_metrics
from preprocessing.image_preprocessor import preprocess_pipeline
from ocr_engines.tesseract_ocr import TesseractEngine
from ocr_engines.easyocr_engine import EasyOCREngine
from ocr_engines.paddleocr_engine import PaddleOCREngine
from ocr_engines.doctr_engine import DocTREngine
from ocr_engines.surya_engine import SuryaEngine


DISTORTION_CONDITIONS = [
    ("original", "standard"),
    ("noisy", "binary"),
    ("blurry", "standard"),
    ("rotated", "standard"),
    ("low_resolution", "low_res"),
]


def run_full_benchmark():
    metadata_path = os.path.join(PROJECT_ROOT, "dataset", "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing dataset metadata at {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"Loaded {len(df_meta)} base samples from metadata.csv across {len(DISTORTION_CONDITIONS)} conditions ({len(df_meta) * len(DISTORTION_CONDITIONS)} total images).")

    # Build all 100 image entries (20 base samples x 5 distortion conditions)
    samples = []
    for condition, prep_mode in DISTORTION_CONDITIONS:
        for _, row in df_meta.iterrows():
            sample_id = str(row["sample_id"])
            image_file = str(row["image_file"])
            gt_file = str(row["ground_truth_file"])
            language = str(row["language"])
            doc_category = str(row["category"])

            abs_img = os.path.join(PROJECT_ROOT, "dataset", condition, image_file)
            abs_gt = os.path.join(PROJECT_ROOT, "dataset", "ground_truth", gt_file)

            if not os.path.exists(abs_img):
                raise FileNotFoundError(f"Missing image: {abs_img}")
            with open(abs_gt, "r", encoding="utf-8") as f:
                gt_text = f.read().strip()

            proc_img = preprocess_pipeline(abs_img, mode=prep_mode)

            samples.append({
                "sample_id": sample_id,
                "image_file": image_file,
                "condition": condition,
                "doc_category": doc_category,
                "language": language,
                "abs_img": abs_img,
                "proc_img": proc_img,
                "gt_text": gt_text,
            })

    engine_factories = [
        ("Tesseract", TesseractEngine),
        ("EasyOCR", EasyOCREngine),
        ("PaddleOCR", PaddleOCREngine),
        ("docTR", DocTREngine),
        ("SuryaOCR", SuryaEngine),
    ]

    all_records = []
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "raw_outputs"), exist_ok=True)
    raw_csv_path = os.path.join(PROJECT_ROOT, "results", "raw_outputs", "benchmark_results.csv")

    for engine_label, factory in engine_factories:
        print(f"\n==========================================")
        print(f"Loading Engine: {engine_label}")
        print(f"==========================================")
        engine = factory()

        # Warm-up inference (excluded from benchmark timings)
        _ = engine.predict(samples[0]["abs_img"], lang=samples[0]["language"])

        for pipeline_type in ["raw", "preprocessed"]:
            desc = f"{engine_label} [{pipeline_type}]"
            for sample in tqdm(samples, desc=desc, ncols=90):
                inp = sample["abs_img"] if pipeline_type == "raw" else sample["proc_img"]
                try:
                    pred = engine.predict(inp, lang=sample["language"])
                    pred_text = pred["text"]
                    latency = pred["latency_sec"]
                    ram_mb = pred["ram_delta_mb"]
                except Exception as exc:
                    pred_text = ""
                    latency = 0.0
                    ram_mb = 0.0
                    print(f"\n[WARN] {engine_label} failed on {sample['condition']}/{sample['image_file']}: {exc}")

                scores = compute_ocr_metrics(
                    reference=sample["gt_text"],
                    hypothesis=pred_text,
                    lang=sample["language"],
                )

                all_records.append({
                    "engine": engine_label,
                    "pipeline": pipeline_type,
                    "sample_id": sample["sample_id"],
                    "image_file": sample["image_file"],
                    "condition": sample["condition"],
                    "doc_category": sample["doc_category"],
                    "language": sample["language"],
                    "ground_truth": sample["gt_text"],
                    "predicted_text": pred_text,
                    "cer": scores["cer"],
                    "wer": scores["wer"],
                    "char_accuracy_pct": scores["char_accuracy_pct"],
                    "word_accuracy_pct": scores["word_accuracy_pct"],
                    "latency_sec": latency,
                    "ram_delta_mb": ram_mb,
                })

        # Save incremental checkpoint after each engine finishes
        pd.DataFrame(all_records).to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved checkpoint ({len(all_records)} rows) -> {raw_csv_path}")

        # Free GPU/CPU memory before loading next engine
        del engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    df_results = pd.DataFrame(all_records)
    summary_path = os.path.join(PROJECT_ROOT, "results", "summary_by_engine.csv")
    summary = (
        df_results.groupby(["engine", "pipeline", "language"], as_index=False)
        .agg(
            avg_cer=("cer", "mean"),
            avg_wer=("wer", "mean"),
            avg_char_acc=("char_accuracy_pct", "mean"),
            avg_word_acc=("word_accuracy_pct", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            avg_ram_mb=("ram_delta_mb", "mean"),
        )
        .round(4)
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n================ BENCHMARK COMPLETE ================")
    print(summary.to_string(index=False))
    print(f"\nFull results saved to: {raw_csv_path}")
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    run_full_benchmark()