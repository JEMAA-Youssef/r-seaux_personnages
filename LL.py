#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LL.py — Extraction ULTRA PRO MAXIMUM des lieux (Tâche 3 AMS)

Ce script génère la liste finale des Lieux et Organisations (LL)
en appliquant un filtrage très strict basé sur des listes blanches
et noires, ainsi que des heuristiques linguistiques.

Fonctionnalités clés :
1. Exclusion des personnages confirmés (LP).
2. Nettoyage et normalisation des n-grammes complexes (ex: "Aurora Monde").
3. Exclusion du bruit sémantique (concepts abstraits, gentilés).
4. Exclusion du bruit grammatical (fragments de phrases, verbes "être", pronoms).
5. Inclusion par heuristique des lieux canoniques et composés (ex: "Secteur Mycogène").

Auteur : [Votre Nom / Binôme]
Date : Novembre 2025
"""

import argparse
import re
from pathlib import Path

# ================================================================
# 1. WHITELIST (Lieux Asimov confirmés)
# ================================================================
# Lieux et Organisations canoniques qui seront acceptés prioritairement.
CANONICAL = {
    "Trantor","Hélicon","Mycogène","Terminus","Aurora","Solaria","Gaia",
    "Comporellon","Smyrno","Kalgan","Siwenna","Anacréon",
    "Dahl","Kan","Streeling","Rossem",
    "New York","Washington","Los Angeles","Berlin","Budapest","Toronto",
    "Canterbury","Norwich","Brighton","Winnipeg","Trenton","Billibotton",
    "Sacratorium","Université de Streeling","Palais","Secteur Mycogène"
}

# ================================================================
# 2. BLACKLIST SÉMANTIQUE
# ================================================================
# Termes à rejeter car ils sont des concepts, du bruit ou des gentilés.
SEMANTIC_NOISE = {
    # Concepts et Abstractions (à éliminer de LL)
    "action","animation","administration","opinion","introduction","instruction",
    "image","existence","analyse","idée","obsession","résultat","impact",
    "logique","raison","invasion","précaution","déduction",

    # Généralités géographiques qui doivent être filtrées par leur nom spécifique
    "monde","mondes","monde extérieur","mondes extérieurs","empire","galaxie",
    "galactique","universités","ville","cité",

    # Dialogue/grammaire et Interjections
    "produit-on","qu'entend-on","qu'on",
    "c'est","c’est","voilà","eh","hein","attention",

    # Noms communs faux positifs (à éliminer)
    "garçon","marron","simpson","grisnuage","soupir"
}

# Peuples (Gentilés) : à éliminer absolument pour ne garder que le Lieu (LOC)
GENTILES = {
    "mycogénien","mycogéniens","spacien","spaciens","terrien","terriens",
    "trantorien","trantoriens","yorkais","billibottains","impériaux","dahlites"
}

# Règles de validation
MONDE_NOISE = {"monde","Monde","MONDE"}  # Termes spécifiques à nettoyer
SUFFIXES = ("or","ia","on","um","polis","grad","ville","town","land")  # Suffixes d'inclusion par heuristique
RE_PROP = re.compile(r"^[A-ZÉÈÀÂÔÎ][A-Za-zéèàçùâêîôû'-]+$")  # Regex pour premier mot capitalisé


# ================================================================
# 3. FONCTIONS UTILITAIRES DE FILTRAGE
# ================================================================
def load_list(path: Path):
    """Charge un fichier texte (LP, L) ligne par ligne."""
    if not path.is_file():
        return []
    return [x.strip() for x in path.open("r",encoding="utf-8") if x.strip()]

def clean_composite_noise(token):
    """
    Normalisation : Simplifie les n-grammes de lieux/orgs complexes en retirant 
    les mots génériques ou les particules.
    Ex: "Aurora Monde" → "Aurora"
    Ex: "Secteur de Mycogène" → "Secteur Mycogène"
    """
    parts = token.split()

    # X Monde → X
    if len(parts) == 2 and parts[1].lower() == "monde":
        return parts[0]

    # X de Y → X Y (gestion des particules courantes de noms d'institutions)
    if len(parts) == 3 and parts[0] in ("Secteur","Université","Palais") and parts[1].lower() == "de":
        return f"{parts[0]} {parts[2]}"

    return token

def contains_verb_etre(s):
    """Vrai si la chaîne contient une forme du verbe 'être' (indicateur de dialogue)."""
    return bool(re.search(r"\b(est|sont|c'est|c’est|était|étaient)\b", s.lower()))

def contains_pronoun(s):
    """Vrai si la chaîne contient un pronom personnel (indicateur de dialogue/bruit)."""
    return bool(re.search(r"\b(je|j'|tu|il|elle|nous|vous|ils|elles|on|moi|toi|lui)\b", s.lower()))

def is_semantic_garbage(s):
    """
    Vérifie si le candidat doit être rejeté à cause d'un contenu sémantique non-lieu.
    """
    parts = s.lower().split()

    # Rejet des concepts seuls
    if s.lower() in SEMANTIC_NOISE:
        return True

    # Rejet des PER + bruit (ex: Baley Or, Hari L'obsession)
    if len(parts) >= 2 and RE_PROP.match(s.split()[0]) and parts[-1] in SEMANTIC_NOISE:
        return True

    # Rejet des fragments contenants le verbe être (même si le filtre principal l'a raté)
    if contains_verb_etre(s):
        return True

    return False

def is_gentile_combo(s):
    """Vérifie les combinaisons Peuple + Lieu/Autre (ex: Terriens Spacetown)."""
    parts = s.lower().split()
    return len(parts) >= 2 and parts[0] in GENTILES

def looks_like_location(token):
    """
    Heuristique d'inclusion : Vérifie si le candidat (après tous les rejets)
    ressemble linguistiquement à un Lieu/Organisation.
    """
    t = token.strip()
    parts = t.split()

    # Inclusion 1 : Whitelist (Priorité maximale)
    if t in CANONICAL:
        return True

    # Rejet des mots qui ne commencent pas par une majuscule
    if not RE_PROP.match(parts[0]):
        return False

    # Inclusion 2 : Suffixes
    if len(parts) == 1 and RE_PROP.match(t):
        if any(t.lower().endswith(s) for s in SUFFIXES):
            return True

    # Inclusion 3 : Composé Institution + Lieu canonique (ex: Secteur Mycogène)
    if len(parts)==2 and parts[0] in ("Secteur","Université","Palais") and parts[1] in CANONICAL:
        return True

    return False


# ================================================================
# 4. FONCTION PRINCIPALE (MAIN)
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Génère la liste LL par filtrage strict.")
    parser.add_argument("--L", type=Path, default="outputs/L.txt", help="Liste brute des candidats.")
    parser.add_argument("--LP", type=Path, default="outputs/LP_final.txt", help="Liste des personnages confirmés (pour l'exclusion).")
    parser.add_argument("-o","--output", type=Path, default="outputs/LL_final.txt", help="Fichier de sortie final (LL.txt).")
    args = parser.parse_args()

    L = load_list(args.L)
    LP = set(load_list(args.LP))  # Liste des PER à exclure

    LL = set()  # Ensemble des lieux retenus

    print("--- Démarrage de la Tâche 3 (Extraction LL) ---")

    for item in L:

        # 1. Exclusion Personnages (Filtre le plus important)
        if item in LP:
            continue

        # 2. Nettoyage et Normalisation
        base = clean_composite_noise(item)

        # --- Chaîne de Rejets Stricts (Les Garde-Fous) ---

        # 3. Rejet Bruit Sémantique (Concepts et Abstractions)
        if is_semantic_garbage(base):
            continue

        # 4. Rejet Combinaison Peuple + Lieu
        if is_gentile_combo(base):
            continue

        # 5. Rejet Pronoms (Bruit conversationnel final)
        if contains_pronoun(base):
            continue

        # --- Inclusion (Heuristique) ---

        # 6. Le candidat nettoyé est-il un Lieu/Org valide ?
        if looks_like_location(base):
            LL.add(base)

    out = sorted(list(LL))

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as f:
        for x in out:
            f.write(x+"\n")

    print(f"✔ Terminé. {len(out)} lieux extraits.")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
