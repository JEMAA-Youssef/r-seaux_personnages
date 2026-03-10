#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
listLP.py — Version "Hardcore" pour nettoyage strict (avec Whitelist pour noms longs).
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Set

# =============================================================================
# 1. LISTE NOIRE ÉTENDUE
# =============================================================================
MANUAL_BLACKLIST = {
    "Trantor", "Siwenna", "Kalgan", "Terminus", "Anacréon", "Smyrno", "Wye", "Cinna",
    "Mycogène", "Mycogénien", "Spacetown", "Aurore", "Solaria", "Gaia", "Rossem", 
    "Neotrantor", "Helicon", "Haven", "Comporellon", "Streeling", "Strelitzia", "Suaverose",
    "Fondation", "Empire", "Galaxie", "Encyclopaedia", "Galactica", "Encyclopaedia Galactica",
    "Spaciens", "Spacien", "Terrien", "Terriens", "Robot", "Robotique", "Positronique",
    "Livre", "Plan", "Toilettes", "Secteur", "Université", "Mairie", "Préfecture", "Bibliothèque",
    "Sacratorium", "Psychohistoire", "Psychohistorien", "Mathématicien", "Projet", "Chapitre",
    "Toilettes", "Cités", "Ville", "Monde", "Mondes", "Extérieur", "Intérieur",
    "Messieurs", "Madame", "Monsieur", "Maître", "Maîtresse", "Docteur", "Maire", 
    "Sœur", "Frère", "Fils", "Père", "Mère", "Lieutenant", "Général", "Commissaire",
    "Oui", "Non", "Bien", "Merci", "Allons", "Allez", "Venez", "Tenez", "Adieu", "Parole",
    "Bonjour", "Bonsoir", "Ciel", "Dieu", "Seigneur", "Zéro", "Primo", "Secundo",
    "Ai-je", "Dit-il", "Réponds-moi", "Comprenez-vous", "Voudriez-vous", "Voyez-vous",
    "Quarante", "Quarante-cinq", "Soixante", "Soixante-douze", "Quarantecinq", "Vingt",
    "Auriez-vous", "Aviez-vous", "Espérez-vous", "Comprenez", "Continuez", "Imaginez",
    "Regardez", "Écoutez", "Ecoutez", "Dis", "Dites", "Dites-moi", "Souvenez-vous",
    "Illico", "Facile", "Exactement", "Absolument", "Agréable", "Difficile", "Désolé",
    "Entendu", "Entrez", "Ouvrez", "Fermez", "Prenez", "Pose", "Marcher", "Grimpons",
    "Doucement", "Vite", "Sérieusement", "Peut-être", "Jamais", "Toujours", "Parfois",
    "Ici", "Là", "Maintenant", "Aujourd'hui", "Demain", "Hier",
    "Cependant", "Pourtant", "Mais", "Car", "Donc", "Or", "Ni", "Ou", "Et",
    "Qui", "Que", "Quoi", "Dont", "Où", "Quand", "Comment", "Pourquoi",
    "Je", "Tu", "Il", "Nous", "Vous", "Ils", "Elles", "On", "Ce", "Ca", "Ça", "C'est",
    "Anciens", "Héliconien", "Kanite", "Médiévaliste", "Médiévalistes",
    "Renégat", "Ridicule", "L'image", "Sire", "L'Empereur", "Randa-là",
    "Galactos"
}

TITLES = [
    "Maire", "Docteur", "Dr", "Monsieur", "M.", "Madame", "Mme",
    "Général", "Empereur", "Lord", "Comte", "Prince", "Roi", "Sœur", "Frère"
]
TITLE_RE = re.compile(r"^(?:" + "|".join(TITLES) + r")\s+[A-Z][a-z]+", re.IGNORECASE)

# --- NOUVEAU : Whitelist pour sauver les héros avec un nom long ---
LONG_NAMES_WHITELIST = {
    "R. Daneel Olivaw", 
    "R. Giskard Reventlov", 
    "Roj Nemennuh Sarton"
}

# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

def load_antidictionary(file_path: Path) -> Set[str]:
    if not file_path.is_file(): return set()
    with file_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

def clean_candidate_string(candidate: str, antidico: Set[str]) -> str:
    parts = candidate.split()
    if not parts: return ""

    parasites = {
        "alors", "mais", "et", "ou", "ni", "car", "donc", "or", "puis", "bref", "comme",
        "quant", "à", "de", "d'", "du", "des", "le", "la", "les", "un", "une", "l'",
        "au", "aux", "en", "dans", "par", "pour", "sur", "avec", "sans", "sous",
        "ce", "cet", "cette", "ces", "celui-ci", "celle-ci", "ceux-ci", "celles-ci",
        "mon", "ton", "son", "notre", "votre", "leur", "nos", "vos", "leurs",
        "quand", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "ça", "c'est",
        "est", "a", "ont", "sont", "était", "avait", "faut", "fait", "dit",
        "j'ai", "j'y", "j'aime", "j'aurais", "j'avais", "qu'est-ce", "n'est-ce",
        "est-ce", "etes-vous", "êtes-vous", "avez-vous", "auriez-vous", "voudriez-vous",
        "comprenez", "comprenez-vous", "écoutez", "regardez", "appelez", "appelez-moi",
        "laissez", "laissez-moi", "allez", "allez-y", "venez", "tenez", "mettez", "mettez-vous",
        "dites", "dites-moi", "souvenez-vous", "imaginez", "voyez-vous", "répondez-moi",
        "pardonnez-moi", "excusez-moi", "noter", "notez",
        "parlez-m'en", "revenons-en", "restons-en", "évitez", "étendez-vous", "étendezvous",
        "espérez-vous", "non", "oui", "bien", "mal", "très", "trop", "ici", "là", "maintenant",
        "toujours", "jamais", "peut-être", "seulement", "aussi", "encore",
        "tout", "tous", "toute", "toutes", "seul", "seule", "même", "autre",
        "pire", "l'habitude", "fuite", "controverse", "d'ailleurs", "d'après", "effectivement", 
        "néanmoins", "toutefois", "garçon", "fille", "homme", "femme", "personne", "gens",
        "maître", "monsieur", "madame", "docteur", "sergent", "mathématicien", "policiers",
        "appelez-moi", "asseyez-vous", "reculez-vous", "redites-moi", "n'allez", "c'était", 
        "assis", "j'aimerais", "redites-moi", "galactica", "encyclopaedia", "mycogène", "trantor",
        "si", "l'un", "l'autre", "c'est-à-dire", "ire", "libre", "trois", "oh", "chut", "règle", 
        "l'obsession", "l'égarement", "couverture", "grimace", "machinchose", "raison", "taciturne", 
        "tambourinant", "quarante-trois", "soixante-douze", "quarante", "quarantecinq", "cinq"
    }

    kept_parts = []
    for p in parts:
        clean_p = re.sub(r"-(vous|nous|moi|toi|le|la|les|lui|leur|y|en)$", "", p, flags=re.IGNORECASE)
        if p.lower() not in parasites and clean_p.lower() not in parasites and p.lower() not in antidico:
            kept_parts.append(p)

    if not kept_parts: return ""
        
    final_parts = [kept_parts[0]]
    for i in range(1, len(kept_parts)):
        if kept_parts[i] != kept_parts[i-1]:
            final_parts.append(kept_parts[i])

    return " ".join(final_parts)

def is_valid_candidate(word: str, label: str, antidico: Set[str]) -> bool:
    clean_word = word.strip()
    blacklist_lower = {w.lower() for w in MANUAL_BLACKLIST}
    
    for w in clean_word.split():
        if w.lower() in blacklist_lower: return False
            
    if clean_word.lower() in antidico: return False
    if re.search(r"[0-9\(\)\[\]]", clean_word): return False
    if clean_word.isupper() and len(clean_word) > 3: return False
    if clean_word.startswith(("'","’")): return False
    
    if label == "PER": return True
    if TITLE_RE.match(clean_word): return True

    return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tâche 2 : Filtrage final LP (Hardcore)")
    parser.add_argument("input_tsv", type=Path, help="Fichier L_tagged.tsv")
    parser.add_argument("-o", "--output", type=Path, default="outputs/LP_final.txt", help="Sortie")
    parser.add_argument("--antidico", type=Path, default="resources/antidictionnaire.txt", help="Antidico")
    args = parser.parse_args()

    print("--- Démarrage Tâche 2 (Mode Strict avec Whitelist) ---")
    antidico = load_antidictionary(args.antidico)
    unique_characters = set()

    with args.input_tsv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) 
        
        for row in reader:
            if len(row) < 3: continue 
            candidate, label = row[0], row[2]
            
            candidate_clean = clean_candidate_string(candidate, antidico)
            if not candidate_clean: continue

            if is_valid_candidate(candidate_clean, label, antidico):
                num_words = len(candidate_clean.split())
                
                # --- CORRECTION ICI : On valide si <= 3 mots OU si c'est dans la whitelist ---
                if len(candidate_clean) > 2 and (num_words <= 3 or candidate_clean in LONG_NAMES_WHITELIST):
                    unique_characters.add(candidate_clean)

    final_list = sorted(list(unique_characters))
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for person in final_list:
            f.write(f"{person}\n")

    print(f"Terminé. {len(final_list)} personnages retenus.")
    print(f"-> {args.output}")

if __name__ == "__main__":
    main()

#python3 listLP.py outputs/L_tagged.tsv -o outputs/LP_final.txt