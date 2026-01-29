#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corpus_stats.py — Calcule les statistiques pour le rapport
"""

import argparse
from pathlib import Path
import re
import sys

def count_words_and_sentences(text):
    # Comptage basique des phrases (basé sur la ponctuation forte)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    # Comptage des mots (basé sur les espaces et la ponctuation)
    words = re.findall(r'\w+', text.lower())
    
    return len(sentences), len(words), set(words)

def count_lines_in_file(file_path, label):
    """Compte les lignes d'un fichier avec gestion d'erreur."""
    if not file_path.exists():
        print(f"- {label:<30} : [FICHIER INTROUVABLE] ({file_path})")
        return
    
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            # On compte les lignes non vides
            count = sum(1 for line in f if line.strip())
        print(f"- {label:<30} : {count}")
    except Exception as e:
        print(f"- {label:<30} : [ERREUR LECTURE] {e}")

def main():
    parser = argparse.ArgumentParser()
    # Dossier du corpus
    parser.add_argument("--corpus", default="corpus_asimov_leaderboard", type=Path)
    
    # Chemins par défaut vers vos fichiers de sortie
    parser.add_argument("--L", default="outputs/L.txt", type=Path)
    parser.add_argument("--LP", default="outputs/LP_final.txt", type=Path)
    parser.add_argument("--LL", default="outputs/LL_final.txt", type=Path)
    
    args = parser.parse_args()

    print("========================================")
    print("       STATISTIQUES DU PROJET           ")
    print("========================================")
    
    # --- 1. VOLUMÉTRIE DU CORPUS ---
    total_files = 0
    total_size = 0
    total_sentences = 0
    total_words = 0
    vocabulaire_global = set()

    # Recherche récursive des fichiers
    files = list(args.corpus.rglob("*.txt.preprocessed"))
    if not files:
        print("Info : Pas de .txt.preprocessed trouvés, recherche des .txt...")
        files = list(args.corpus.rglob("*.txt"))

    if not files:
        print(f"ERREUR : Aucun fichier trouvé dans le dossier '{args.corpus}'")
        sys.exit(1)

    print(f"Analyse de {len(files)} fichiers en cours...")

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            
            # Stats fichiers
            total_files += 1
            total_size += f.stat().st_size
            
            # Stats linguistiques
            nb_sent, nb_words, vocab = count_words_and_sentences(text)
            total_sentences += nb_sent
            total_words += nb_words
            vocabulaire_global.update(vocab)
        except Exception as e:
            print(f"Erreur lecture fichier {f.name}: {e}")

    print("\n1. STATISTIQUES DU CORPUS (Données brutes)")
    print(f"- Nombre de chapitres (fichiers) : {total_files}")
    print(f"- Taille totale sur disque       : {total_size / 1024:.2f} Ko")
    print(f"- Nombre total de phrases        : {total_sentences}")
    print(f"- Nombre total de mots (Tokens)  : {total_words}")
    print(f"- Vocabulaire unique             : {len(vocabulaire_global)}")

    # --- 2. RÉSULTATS DES LISTES ---
    print("\n2. RÉSULTATS D'EXTRACTION (Tâches 1, 2, 3)")
    
    # Appel de la fonction sécurisée pour chaque fichier
    count_lines_in_file(args.L, "Candidats bruts (Liste L)")
    count_lines_in_file(args.LP, "Personnages (Liste LP)")
    count_lines_in_file(args.LL, "Lieux (Liste LL)")

    print("\n========================================")

if __name__ == "__main__":
    main()