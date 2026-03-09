#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import unicodedata
import re
from collections import Counter
from pathlib import Path

# --- Fonctions de base (reprises de votre script) ---
_LET = r"A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"
TOKEN_RE = re.compile(rf"[{_LET}]+(?:['’][{_LET}]+|-[{_LET}]+)*")

def normalize(text: str) -> str:
    """Nettoie et homogénéise le texte."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("`", "'").replace("´", "'")
    t = t.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    t = re.sub(r"-\s*\n\s*", "", t)
    t = t.replace("\u00A0", " ").replace("\u202F", " ").replace("\t", " ") # 10 € & bonjour !
    t = re.sub(r"\n{2,}", "\n", t)#espace multiple vertical
    t = re.sub(r"[ ]{2,}", " ", t)#espace multiple horizontal
    return t

def tokenize(text: str) -> list[str]:
    """Sépare le texte en tokens."""
    return TOKEN_RE.findall(text)

def is_capitalized(tok: str) -> bool:
    """Vérifie si un token commence par une majuscule."""
    return tok and tok[0].isupper()

# --- Logique principale de la TÂCHE 1 ---

def build_raw_list_L(text: str, max_n: int = 3) -> Counter:
    """
    Construit la liste L 'en vrac et sans filtrage' (Tâche 1).
    
    L'unique heuristique est de collecter les n-grammes (n=1, 2, 3)
    qui commencent par un mot capitalisé, comme suggéré par le TP[cite: 27, 29].
    """
    tokens = tokenize(text)
    counts = Counter()
    n = len(tokens)
    
    print(f"Tokenisation terminée, {n} tokens trouvés. Scan des n-grammes...")
    
    for i in range(n):
        # La seule règle de candidature : le premier mot doit être capitalisé.
        if is_capitalized(tokens[i]):
            
            # 1. Ajouter le unigram (n=1)
            unigram = tokens[i]
            counts[unigram] += 1
            
            # 2. Ajouter le bigram (n=2)
            if i + 1 < n:
                bigram = f"{tokens[i]} {tokens[i+1]}"
                counts[bigram] += 1
                
            # 3. Ajouter le trigram (n=3)
            if i + 2 < n:
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                counts[trigram] += 1
                
    return counts

# --- Exécution ---
def main():
    parser = argparse.ArgumentParser(description="Tâche 1: Construit la liste L BRUTE (n=1,2,3) ")
    parser.add_argument("texts", nargs="+", help="Un ou plusieurs fichiers .txt d'entrée.")
    parser.add_argument("-o", "--outdir", default=".", help="Dossier de sortie.")
    args = parser.parse_args()

    print("Lecture et normalisation des fichiers corpus...")
    text_parts = [Path(p).read_text(encoding="utf-8") for p in args.texts]
    raw_text = "\n".join(text_parts)
    full_text = normalize(raw_text)

    print("Construction de la liste L brute (n=1,2,3)...")
    counts = build_raw_list_L(full_text, max_n=3)

    # Écriture des fichiers de sortie
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Trier par fréquence (décroissant) puis par ordre alphabétique
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    # Nommer les fichiers "L_brute" pour les différencier
    out_txt_path = outdir / "L_brute.txt"
    out_tsv_path = outdir / "L_brute_counts.tsv"

    with out_txt_path.open("w", encoding="utf-8") as f:
        for k, _ in items:
            f.write(f"{k}\n")
            
    with out_tsv_path.open("w", encoding="utf-8") as f:
        for k, c in items:
            f.write(f"{k}\t{c}\n")

    print(f"\nTerminé. Liste L brute générée avec {len(counts)} candidats.")
    print(f"-> {out_txt_path}")
    print(f"-> {out_tsv_path}")

if __name__ == "__main__":
    main()