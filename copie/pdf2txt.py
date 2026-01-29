#!/usr/bin/env python3
"""
Convert a (possibly scanned) French PDF to UTF-8 text using a robust OCR pipeline.

Outputs:
  - <pdf>.txt   : extracted plain text (UTF-8)

Dependencies (Ubuntu/WSL):
  sudo apt update
  sudo apt install -y ocrmypdf tesseract-ocr tesseract-ocr-fra poppler-utils
  pip install ftfy
"""

import argparse
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

# -------------------------
# Optional Unicode fix-ups
# -------------------------
try:
    from ftfy import fix_text
except ImportError:
    def fix_text(s: str) -> str:
        return s

# -------------------------
# System helpers
# -------------------------
def run(cmd, check=True, capture=False, text=True):
    return subprocess.run(cmd, check=check, capture_output=capture, text=text)

def have(cmd_name: str) -> bool:
    return shutil.which(cmd_name) is not None

# -------------------------
# Text normalization
# -------------------------
def clean_text(raw: str) -> str:
    txt = fix_text(raw)
    txt = unicodedata.normalize("NFC", txt)
    return txt.replace("\ufffd", "")

# -------------------------
# OCR pipeline
# -------------------------
def ocrmypdf_then_pdftotext(pdf_path: Path) -> str:
    if not have("ocrmypdf") or not have("pdftotext"):
        raise RuntimeError("ocrmypdf or pdftotext not available")
    with tempfile.TemporaryDirectory() as tmpd:
        ocr_pdf = Path(tmpd) / "ocr.pdf"
        cmd_ocr = [
            "ocrmypdf",
            "--force-ocr",
            "--language", "fra",
            "--optimize", "0",
            "--output-type", "pdf",
            str(pdf_path),
            str(ocr_pdf),
        ]
        print("Running OCR with ocrmypdf (language=fra)...")
        run(cmd_ocr)

        print("Extracting text with pdftotext...")
        res = run(["pdftotext", "-enc", "UTF-8", str(ocr_pdf), "-"], capture=True)
        return res.stdout

def fallback_tesseract_per_page(pdf_path: Path, dpi: int = 300) -> str:
    if not have("pdftoppm") or not have("tesseract"):
        raise RuntimeError("pdftoppm or tesseract not available")
    text_parts = []
    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        prefix = tmpd / "page"
        print("Converting PDF to images with pdftoppm...")
        run(["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)])

        pages = sorted(tmpd.glob("page-*.png"))
        if not pages:
            raise RuntimeError("No pages produced by pdftoppm (corrupt PDF?).")

        print(f"OCR {len(pages)} page(s) with tesseract (fra)...")
        for i, img in enumerate(pages, start=1):
            res = run(["tesseract", str(img), "stdout", "-l", "fra"], capture=True)
            page_text = res.stdout.strip()
            text_parts.append(page_text)
            text_parts.append(f"\n\n——— [PAGE {i}] ———\n\n")
    return "".join(text_parts)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="French PDF → UTF-8 text (robust OCR).")
    ap.add_argument("pdf", help="Input PDF path")
    ap.add_argument(
        "-o", "--output",
        help="Output .txt path (default: same base name as the PDF, .txt)",
        default=None,
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"Input file not found: {pdf_path}")

    out_txt_path = Path(args.output).expanduser().resolve() if args.output else pdf_path.with_suffix(".txt")

    # OCR pipeline: preferred path, then fallback
    try:
        extracted = ocrmypdf_then_pdftotext(pdf_path)
    except Exception as e:
        print(f"Warning: ocrmypdf/pdftotext failed ({e}). Falling back to per-page tesseract OCR...")
        extracted = fallback_tesseract_per_page(pdf_path)

    text = clean_text(extracted)

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(text, encoding="utf-8")
    print(f"Text saved to: {out_txt_path}")
    print("Done.")

if __name__ == "__main__":
    main()



    #python3 pdf2txt.py corpus_asimov/Fondation_et_empire_sample.pdf
