#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import spacy
from pathlib import Path
from typing import List, Set

# --- Configuration & Patrons (No changes here) ---
TITLES = [
    "Maire", "Docteur", "Dr", "Monsieur", "M.", "Madame", "Mme",
    "Général", "Empereur", "Lord", "Comte", "Prince", "Roi"
]
TITLE_PATTERN = re.compile(
    r"^(?i)(?:" + "|".join(TITLES) + r")\s+([A-Z]\w+)$"
)


# --- Fonctions de chargement (No changes here) ---

def load_word_list(file_path: Path) -> List[str]:
    """Loads a list of terms from a text file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_antidictionary(file_path: Path) -> Set[str]:
    """Loads an antidictionary into a set for optimal performance."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Antidictionary not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_full_corpus(corpus_dir: Path) -> str:
    """Loads and concatenates all preprocessed text files from the corpus directory."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    files = sorted(list(corpus_dir.glob("*/*.txt.preprocessed")))
    if not files:
        raise FileNotFoundError(f"No .txt.preprocessed files found in subdirectories of {corpus_dir}")
    text_parts = [f.read_text(encoding="utf-8") for f in files]
    return "\n".join(text_parts)


# --- Fonctions de filtrage (apply_ner_filter is now corrected) ---

def apply_antidictionary_filter(candidates: List[str], antidico: Set[str]) -> List[str]:
    """Applies the antidictionary filter to a list of candidates."""
    kept_candidates = []
    for candidate in candidates:
        last_word = candidate.split()[-1].lower()
        if last_word not in antidico:
            kept_candidates.append(candidate)
    return kept_candidates


def apply_pattern_filter(candidates: List[str]) -> List[str]:
    """Identifies candidates that match character name patterns (e.g., Title + Name)."""
    identified_characters = set()
    for candidate in candidates:
        match = TITLE_PATTERN.match(candidate)
        if match:
            identified_characters.add(candidate)
            identified_characters.add(match.group(1))  # Add the captured name as well
    return sorted(list(identified_characters))


# --- MODIFIED AND CORRECTED FUNCTION ---
def apply_ner_filter(candidates: List[str], full_text: str, nlp_model) -> List[str]:
    """
    Validates candidates using a STRICT check against spaCy's NER person entities.
    """
    print("      -> Analyzing text with spaCy NER (this may take a moment)...")
    doc = nlp_model(full_text)
    persons_in_text = {ent.text for ent in doc.ents if ent.label_ == "PER"}
    print(f"      -> Found {len(persons_in_text)} unique person entities.")

    validated_candidates = []
    # The new, stricter logic:
    # A candidate is only kept if the candidate string ITSELF is in the set
    # of persons identified by spaCy.
    for candidate in candidates:
        if candidate in persons_in_text:
            validated_candidates.append(candidate)

    return sorted(list(set(validated_candidates)))


# --- Main orchestrator (No changes here) ---

def main():
    parser = argparse.ArgumentParser(description="Filters a candidate list to produce the final LP list.")
    parser.add_argument("--candidates", type=Path, default="L.txt", help="Path to the L.txt candidate file.")
    parser.add_argument("--corpus", type=Path, default="corpus_asimov_leaderboard",
                        help="Path to the corpus directory.")
    parser.add_argument("--antidico", type=Path, default="resources/antidictionnaire.txt",
                      help="Path to the antidictionary file.")
    parser.add_argument("-o", "--output", type=Path, default="LP_final.txt", help="Output file for the final LP list.")
    args = parser.parse_args()

    print("--- Starting Full LP List Generation Pipeline ---")

    print("\n[Étape 1] Loading data...")
    candidates_L = load_word_list(args.candidates)
    antidictionary = load_antidictionary(args.antidico)
    full_corpus = load_full_corpus(args.corpus)
    print(f"   - Loaded {len(candidates_L)} candidates from {args.candidates}.")

    print("\n[Étape 2] Applying Antidictionary Filter...")
    filtered_step1 = apply_antidictionary_filter(candidates_L, antidictionary)
    print(f"   - {len(filtered_step1)} candidates remaining.")

    print("\n[Étape 3] Applying Pattern Filter...")
    candidates_from_patterns = apply_pattern_filter(filtered_step1)
    print(f"   - Found {len(candidates_from_patterns)} candidates matching title patterns.")

    print("\n[Étape 4] Combining candidate lists...")
    combined_candidates = sorted(list(set(filtered_step1 + candidates_from_patterns)))
    print(f"   - Total of {len(combined_candidates)} unique candidates to be validated.")

    print("\n[Étape 5] Final validation with spaCy NER...")
    print("   -> Loading spaCy model...")
    nlp = spacy.load("fr_core_news_sm")
    final_list_LP = apply_ner_filter(combined_candidates, full_corpus, nlp)

    print(f"\n[FINISH] Saving {len(final_list_LP)} final characters to {args.output}...")
    with args.output.open("w", encoding="utf-8") as f:
        for item in final_list_LP:
            f.write(f"{item}\n")

    print("\n--- Pipeline finished successfully! ---")


if __name__ == "__main__":
    main()