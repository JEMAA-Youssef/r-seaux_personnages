#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
listLP.py — Version "Hardcore" pour nettoyage strict.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Set

# =============================================================================
# 1. LISTE NOIRE ÉTENDUE (Basée sur vos résultats)
# =============================================================================
# Ces mots, s'ils sont trouvés, disqualifient immédiatement le candidat
MANUAL_BLACKLIST = {
    # Lieux & Concepts
    "Trantor", "Siwenna", "Kalgan", "Terminus", "Anacréon", "Smyrno", "Wye", "Cinna",
    "Mycogène", "Mycogénien", "Spacetown", "Aurore", "Solaria", "Gaia", "Rossem", 
    "Neotrantor", "Helicon", "Haven", "Comporellon", "Streeling", "Strelitzia", "Suaverose",
    "Fondation", "Empire", "Galaxie", "Encyclopaedia", "Galactica", "Encyclopaedia Galactica",
    "Spaciens", "Spacien", "Terrien", "Terriens", "Robot", "Robotique", "Positronique",
    "Livre", "Plan", "Toilettes", "Secteur", "Université", "Mairie", "Préfecture", "Bibliothèque",
    "Sacratorium", "Psychohistoire", "Psychohistorien", "Mathématicien", "Projet", "Chapitre",
    "Toilettes", "Cités", "Ville", "Monde", "Mondes", "Extérieur", "Intérieur",
    
    # Titres génériques (seuls)
    "Messieurs", "Madame", "Monsieur", "Maître", "Maîtresse", "Docteur", "Maire", 
    "Sœur", "Frère", "Fils", "Père", "Mère", "Lieutenant", "Général", "Commissaire",
    
    # Verbes et Mots parasites (Le gros du nettoyage)
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

# Titres autorisés pour le rattrapage
TITLES = [
    "Maire", "Docteur", "Dr", "Monsieur", "M.", "Madame", "Mme",
    "Général", "Empereur", "Lord", "Comte", "Prince", "Roi", "Sœur", "Frère"
]
TITLE_RE = re.compile(r"^(?:" + "|".join(TITLES) + r")\s+[A-Z][a-z]+", re.IGNORECASE)

# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

def load_antidictionary(file_path: Path) -> Set[str]:
    if not file_path.is_file(): return set()
    with file_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}



def clean_candidate_string(candidate: str, antidico: Set[str]) -> str:
    """
    Nettoie un candidat en supprimant TOUS les mots parasites, où qu'ils soient.
    """
    parts = candidate.split()
    if not parts: return ""

    # 1. Liste complète des parasites (tout en minuscules !)
    parasites = {
            # Mots de liaison & Prépositions
            "alors", "mais", "et", "ou", "ni", "car", "donc", "or", "puis", "bref", "comme",
            "quant", "à", "de", "d'", "du", "des", "le", "la", "les", "un", "une", "l'",
            "au", "aux", "en", "dans", "par", "pour", "sur", "avec", "sans", "sous",
            "ce", "cet", "cette", "ces", "celui-ci", "celle-ci", "ceux-ci", "celles-ci",
            "mon", "ton", "son", "notre", "votre", "leur", "nos", "vos", "leurs",
            "quand", # Ajouté : "Quand Hari Seldon"
            
            # Verbes & Pronoms (Bruit conversationnel)
            "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "ça", "c'est",
            "est", "a", "ont", "sont", "était", "avait", "faut", "fait", "dit",
            "j'ai", "j'y", "j'aime", "j'aurais", "j'avais", "qu'est-ce", "n'est-ce",
            "est-ce", "etes-vous", "êtes-vous", "avez-vous", "auriez-vous", "voudriez-vous",
            "comprenez", "comprenez-vous", "écoutez", "regardez", "appelez", "appelez-moi",
            "laissez", "laissez-moi", "allez", "allez-y", "venez", "tenez", "mettez", "mettez-vous",
            "dites", "dites-moi", "souvenez-vous", "imaginez", "voyez-vous", "répondez-moi",
            "pardonnez-moi", "excusez-moi", "noter", "notez",
            "parlez-m'en", "revenons-en", "restons-en", "évitez", "étendez-vous", "étendezvous", # Ajoutés
            "espérez-vous", # Ajouté
            
            # Adverbes & Autres
            "non", "oui", "bien", "mal", "très", "trop", "ici", "là", "maintenant",
            "toujours", "jamais", "peut-être", "seulement", "aussi", "encore",
            "tout", "tous", "toute", "toutes", "seul", "seule", "même", "autre",
            "pire", "l'habitude", "fuite", "controverse", # Ajoutés
            
            # Mots spécifiques & Titres
            "d'ailleurs", "d'après", "effectivement", "néanmoins", "toutefois",
            "garçon", "fille", "homme", "femme", "personne", "gens",
            "maître", "monsieur", "madame", "docteur", "sergent",
            "mathématicien", "policiers", # Ajoutés
            
            # Fragments conversationnels déjà identifiés
            "appelez-moi", "asseyez-vous", "reculez-vous", "redites-moi", 
            "n'allez", "c'était", "assis", "j'aimerais", "redites-moi",
            
            # Lieux/Concepts parasites (en minuscules pour être sûr)
            "galactica", "encyclopaedia", "mycogène", "trantor",
            
            # Mots de liaison, adverbes et petits mots supplémentaires
            "si", "l'un", "l'autre", "c'est-à-dire", "ire", "libre", "trois", 
            "oh", "chut", "règle", "l'obsession", "l'égarement", "couverture", 
            "grimace", "machinchose", "raison", "taciturne", "tambourinant",
            "quarante-trois", "soixante-douze", "quarante", "quarantecinq", "cinq" # Ajoutés (Chiffres)
        }

        # 2. Nettoyage INTERNE : On garde uniquement les mots qui NE SONT PAS des parasites
        # Cela va transformer "Davan Appelez-moi Davan" en "Davan Davan"
    kept_parts = []
    for p in parts:
        # On enlève les suffixes gênants (-vous, -moi)
        clean_p = re.sub(r"-(vous|nous|moi|toi|le|la|les|lui|leur|y|en)$", "", p, flags=re.IGNORECASE)
            
        # On vérifie si le mot (ou sa version nettoyée) est un parasite
        if p.lower() not in parasites and clean_p.lower() not in parasites and p.lower() not in antidico:
            kept_parts.append(p)

        # 3. Dédoublonnage simple (Davan Davan -> Davan)
    if not kept_parts: return ""
        
    final_parts = [kept_parts[0]]
    for i in range(1, len(kept_parts)):
        if kept_parts[i] != kept_parts[i-1]: # Si différent du précédent
            final_parts.append(kept_parts[i])

    return " ".join(final_parts)

def is_valid_candidate(word: str, label: str, antidico: Set[str]) -> bool:
    clean_word = word.strip()
    
    # 1. Vérification mot par mot contre la Blacklist (pour attraper ENCYCLOPAEDIA GALACTICA)
    # On crée une version minuscule de la blacklist pour être sûr
    blacklist_lower = {w.lower() for w in MANUAL_BLACKLIST}
    
    for w in clean_word.split():
        if w.lower() in blacklist_lower:
            return False
            
    # 2. Filtres classiques
    if clean_word.lower() in antidico: return False
    if re.search(r"[0-9\(\)\[\]]", clean_word): return False
    
    # 3. Refus si tout majuscule et long (ex: GALACTICA) sauf si c'est un sigle court
    if clean_word.isupper() and len(clean_word) > 3: return False

    # 4. Refus si commence par apostrophe
    if clean_word.startswith(("'","’")): return False

    # 5. Acceptation
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

    print("--- Démarrage Tâche 2 (Mode Strict) ---")
    antidico = load_antidictionary(args.antidico)
    
    # On utilise un dictionnaire pour garder la version la plus courte d'un nom
    # Ex: Si on a "Seldon" et "Seldon Amaryl", on garde les deux.
    # Mais si on a "Baley" et "Baley Mais", le nettoyage de "Baley Mais" donne "Baley".
    unique_characters = set()

    with args.input_tsv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) 
        
        for row in reader:
            if len(row) < 3: continue 
            candidate, label = row[0], row[2]
            
            # 1. Nettoyage PREALABLE (avant validation)
            # Cela transforme "Baley Est-ce" en "Baley" AVANT de vérifier si c'est valide
            candidate_clean = clean_candidate_string(candidate, antidico)
            
            if not candidate_clean: continue

            # 2. Validation
            if is_valid_candidate(candidate_clean, label, antidico):
                
                # 3. Filtre de Longueur STRICT
                # Un nom > 3 mots est presque toujours une erreur dans ce corpus
                # (Ex: "Policiers RAYCH D'après Hari")
                num_words = len(candidate_clean.split())
                
                if len(candidate_clean) > 2 and num_words <= 3:
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