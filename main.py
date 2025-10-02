#!/usr/bin/env python3
"""
Convert a (possibly scanned) French PDF to UTF-8 text using a robust OCR pipeline,
then produce the raw candidate named-entity list L (unigrams + multiword sequences),
as requested in Task 1 (bulk, unfiltered).

Outputs:
  - <pdf>.txt         : extracted plain text
  - L.txt             : unique candidate strings, sorted by frequency desc, then alphabetically
  - L_counts.tsv      : candidate counts (TSV: candidate<TAB>count)

Dependencies (Ubuntu/WSL):
  sudo apt update
  sudo apt install -y ocrmypdf tesseract-ocr tesseract-ocr-fra poppler-utils
  pip install ftfy
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
import re

# -------------------------
# Optional Unicode fix-ups
# -------------------------
try:
    from ftfy import fix_text
except ImportError:
    # Soft fallback if ftfy is not installed
    def fix_text(s: str) -> str:
        return s


# -------------------------
# System helpers
# -------------------------
def run(cmd, check=True, capture=False, text=True):
    """Run a subprocess command with convenient defaults."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=text
    )

def have(cmd_name: str) -> bool:
    """Return True if an executable is available in PATH."""
    return shutil.which(cmd_name) is not None


# -------------------------
# Text normalization
# -------------------------
def clean_text(raw: str) -> str:
    """Fix common Unicode issues and normalize to NFC."""
    txt = fix_text(raw)
    txt = unicodedata.normalize("NFC", txt)
    # Remove U+FFFD replacement characters if any remain
    txt = txt.replace("\ufffd", "")
    return txt


# -------------------------
# OCR pipeline
# -------------------------
def ocrmypdf_then_pdftotext(pdf_path: Path) -> str:
    """
    Preferred method: OCRMyPDF (fra) -> pdftotext (UTF-8).
    Returns extracted text.
    """
    if not have("ocrmypdf") or not have("pdftotext"):
        raise RuntimeError("ocrmypdf or pdftotext not available")
    with tempfile.TemporaryDirectory() as tmpd:
        ocr_pdf = Path(tmpd) / "ocr.pdf"
        cmd_ocr = [
            "ocrmypdf",
            "--force-ocr",
            "--language", "fra",
            "--optimize", "0",     # for quality use 3 (slower), 0 is faster
            "--output-type", "pdf",
            str(pdf_path),
            str(ocr_pdf)
        ]
        print("Running OCR with ocrmypdf (language=fra)...")
        run(cmd_ocr)

        print("Extracting text with pdftotext...")
        # Emit text to stdout ("-") so we can capture it directly
        res = run(["pdftotext", "-enc", "UTF-8", str(ocr_pdf), "-"], capture=True)
        return res.stdout

def fallback_tesseract_per_page(pdf_path: Path, dpi: int = 300) -> str:
    """
    Fallback if OCRMyPDF is unavailable:
      - pdftoppm -> PNG pages (300 dpi)
      - tesseract -l fra -> text
    Concatenates the text from all pages with mild separators.
    """
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
# Candidate extraction (Task 1 - L)
# -------------------------
# Unicode-aware token pattern: letters (incl. accents), allows internal apostrophes/hyphens.
# Examples captured: "Hari", "Seldon", "Seconde", "Fondation", "Saint-Loup", "d'Artagnan".
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+)*")

def is_capitalized(token: str) -> bool:
    """
    Return True if token starts with an uppercase letter or is all-caps with length >= 2.
    This intentionally includes noise (e.g., chapter numerals mis-OCR'ed) to keep L "in bulk".
    """
    if not token:
        return False
    # Any leading uppercase char
    if token[0].isupper():
        return True
    # All-caps short tokens (e.g., acronyms)
    return token.isupper() and len(token) >= 2

def tokenize(text: str) -> list[str]:
    """Tokenize text into a list of Unicode-aware word tokens."""
    return TOKEN_RE.findall(text)

def extract_L_candidates(text: str, max_ngram: int = 3) -> Counter:
    """
    Build a raw, unfiltered candidate list L from text.
    Strategy:
      - Unigrams: capitalized or all-caps tokens.
      - Multiword sequences (bigrams/trigrams): consecutive capitalized tokens, up to max_ngram.
    Returns a Counter of candidates -> frequency.
    """
    tokens = tokenize(text)

    # Keep an auxiliary list marking whether each token is "capitalized" by our heuristic
    caps_mask = [is_capitalized(t) for t in tokens]

    counts = Counter()

    # Unigrams
    for i, (tok, is_cap) in enumerate(zip(tokens, caps_mask)):
        if is_cap:
            counts[tok] += 1

    # Bigrams and trigrams composed of consecutive "capitalized" tokens
    n = len(tokens)
    for i in range(n):
        if not caps_mask[i]:
            continue
        # Grow sequences up to max_ngram
        seq = [tokens[i]]
        for j in range(i + 1, min(i + max_ngram, n)):
            if not caps_mask[j]:
                break
            seq.append(tokens[j])
            phrase = " ".join(seq)
            counts[phrase] += 1

    return counts

def write_L_outputs(counts: Counter, out_dir: Path):
    """
    Write:
      - L.txt          : unique candidates, sorted by frequency desc then lexicographically
      - L_counts.tsv   : candidate<TAB>count (same order)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by count desc, then by candidate for stability
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    l_txt = out_dir / "L.txt"
    l_counts = out_dir / "L_counts.tsv"

    with open(l_txt, "w", encoding="utf-8") as f:
        for cand, _ in items:
            f.write(f"{cand}\n")

    with open(l_counts, "w", encoding="utf-8") as f:
        for cand, c in items:
            f.write(f"{cand}\t{c}\n")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert a scanned French PDF to UTF-8 text (robust OCR) and produce raw candidate list L."
    )
    parser.add_argument(
        "pdf",
        help="Input PDF path (e.g., corpus_asimov/Fondation_et_empire_sample.pdf)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .txt path (default: same base name as the PDF, .txt)",
        default=None
    )
    parser.add_argument(
        "--no-L",
        action="store_true",
        help="Do not compute the L candidates (only produce the .txt text file)."
    )
    parser.add_argument(
        "--l-outdir",
        default=".",
        help="Directory to write L.txt and L_counts.tsv (default: current directory)."
    )
    args = parser.parse_args()

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

    # Normalize and fix text
    text = clean_text(extracted)

    # Persist .txt
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text saved to: {out_txt_path}")

    # Produce raw candidate list L (Task 1)
    if not args.no_L:
        counts = extract_L_candidates(text, max_ngram=3)
        l_outdir = Path(args.l_outdir).expanduser().resolve()
        write_L_outputs(counts, l_outdir)
        print(f"L outputs written in: {l_outdir} (L.txt, L_counts.tsv)")

    print("Done.")

if __name__ == "__main__":
    main()



# Basic: produce <pdf>.txt + L.txt + L_counts.tsv in current directory
#python3 main.py corpus_asimov/Fondation_et_empire_sample.pdf

# Choose output text path and where to write L files
#python3 main.py corpus_asimov/Fondation_et_empire_sample.pdf \
#  -o outputs/Fondation_et_empire_sample.txt \
#  --l-outdir outputs/

# If you want only the text (no L yet)
#python3 main.py corpus_asimov/Fondation_et_empire_sample.pdf --no-L