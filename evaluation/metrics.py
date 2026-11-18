import re
import unicodedata
import jiwer


def normalize_text(text: str, lang: str = "English", lowercase: bool = False) -> str:
    """
    Normalize ground-truth and OCR hypothesis text fairly across English and Hindi.
    Applies Unicode NFC normalization (crucial for Devanagari matras) and collapses whitespace.
    """
    if text is None:
        return ""
    # Canonical Unicode normalization for Devanagari & Latin scripts
    cleaned = unicodedata.normalize("NFC", str(text))
    # Collapse multiple spaces, tabs, and newlines into single spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if lowercase and lang.lower() == "english":
        cleaned = cleaned.lower()
    return cleaned


def compute_ocr_metrics(reference: str, hypothesis: str, lang: str = "English") -> dict:
    """
    Compute CER, WER, Character Accuracy (%), and Word Accuracy (%) between
    ground-truth `reference` and OCR output `hypothesis`.
    """
    ref = normalize_text(reference, lang=lang, lowercase=False)
    hyp = normalize_text(hypothesis, lang=lang, lowercase=False)

    if not ref:
        cer_val = 0.0 if not hyp else 1.0
        wer_val = 0.0 if not hyp else 1.0
    elif not hyp:
        cer_val = 1.0
        wer_val = 1.0
    else:
        cer_val = float(jiwer.cer(ref, hyp))
        wer_val = float(jiwer.wer(ref, hyp))

    char_acc = max(0.0, (1.0 - cer_val) * 100.0)
    word_acc = max(0.0, (1.0 - wer_val) * 100.0)

    return {
        "cer": round(cer_val, 4),
        "wer": round(wer_val, 4),
        "char_accuracy_pct": round(char_acc, 2),
        "word_accuracy_pct": round(word_acc, 2),
    }


if __name__ == "__main__":
    # Self-test on English and Hindi sample pairs
    eng_gt = "Optical Character Recognition converts scanned documents into searchable digital text."
    eng_pred_exact = "Optical Character Recognition converts scanned documents into searchable digital text."
    eng_pred_minor = "Optical Character Recognition converts scanned document into searchable digital text."

    hin_gt = "ऑप्टिकल कैरेक्टर रिकग्निशन तकनीक मुद्रित दस्तावेजों को डिजिटल पाठ में बदलती है।"
    hin_pred_exact = "ऑप्टिकल कैरेक्टर रिकग्निशन तकनीक मुद्रित दस्तावेजों को डिजिटल पाठ में बदलती है।"
    hin_pred_hallucinated = "Wop/pPhlle CRCDCR A RIAAR dob-ic YGra InpPA2 s sulea 9lh PDPpH oW"

    print("--- Testing Evaluation Metrics Module ---")
    print("1. English Exact Match:     ", compute_ocr_metrics(eng_gt, eng_pred_exact, lang="English"))
    print("2. English 1-Char Typo:     ", compute_ocr_metrics(eng_gt, eng_pred_minor, lang="English"))
    print("3. Hindi Exact Match:       ", compute_ocr_metrics(hin_gt, hin_pred_exact, lang="Hindi"))
    print("4. Hindi Script Mismatch:   ", compute_ocr_metrics(hin_gt, hin_pred_hallucinated, lang="Hindi"))