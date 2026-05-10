#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LL.py — Extraction ULTRA PRO MAXIMUM des lieux (Tâche 3 AMS)
"""

import argparse
import re
from pathlib import Path


CANONICAL = {
    "Trantor","TRANTOR","Hélicon","Mycogène","Terminus","Dahl","Kan","Aurora","Solaria","Gaia",
    "Terre","Billibotton","Sacratorium","SACRATORIUM","Comporellon","Smyrno","Kalgan","Siwenna",
    "Streeling","Rossem","Spacetown","Anacréon",
    "New York","Washington","Los Angeles","Berlin","Budapest","Toronto","Canterbury","Norwich",
    "Brighton","Winnipeg","Trenton","Londres","Newark","Williamsburg","Philadelphie","Shanghai",
    "Tachkent","Buenos Aires","Bronx","Caire",
    "Long Island","New Jersey","Amérique","Alleghanis",
    "Palais","Empire","Capitale","Centrale","L'Empire","L'Université",
    "Mercure","Saturne",
    "Mondes Extérieurs","ENCYCLOPAEDIA GALACTICA","GALACTICA"
}


# Termes à rejeter car ils sont des concepts, du bruit ou des gentilés.
SEMANTIC_NOISE = {
   
    "action","animation","administration","opinion","introduction","instruction",
    "image","existence","analyse","idée","obsession","résultat","impact",
    "logique","raison","invasion","précaution","déduction","opération",
    "question","surpopulation","attention",

    
    "monde","mondes","monde extérieur","galaxie",
    "galactique","universités","ville","cité",

    # Dialogue/grammaire et Interjections
    "produit-on","qu'entend-on","qu'on",
    "c'est","c'est","voilà","eh","hein",

    # Noms communs faux positifs 
    "garçon","simpson","grisnuage","soupir",
    
    "town","torium","angeles","york","jersey"
}


GENTILES = {
    "mycogénien","mycogéniens","yorkais","billibottains","dahlites"
}


GENTILES_LIEUX = {
    "Spacien","Spaciens","Terrien","Terriens","Trantorien","Trantoriens",
    "Impériaux","Extérieurs"
}

# Règles de validation
MONDE_NOISE = {"monde","Monde","MONDE"}  
SUFFIXES = ("or","ia","on","um","polis","grad","ville","town","land")  
RE_PROP = re.compile(r"^[A-ZÉÈÀÂÔÎ][A-Za-zéèàçùâêîôû'-]+$")  


# FONCTIONS UTILITAIRES DE FILTRAGE

def load_list(path: Path):
    """Charge un fichier texte (LP, L) ligne par ligne."""
    if not path.is_file():
        return []
    return [x.strip() for x in path.open("r",encoding="utf-8") if x.strip()]

def normalize_articles(token):
    """
    Normalise les articles contractés pour le matching avec SEMANTIC_NOISE.
    Ex: "L'action" → "action"
    """
    if token.startswith(("L'", "l'", "d'", "D'")):  
        return token[2:]
    return token

def clean_composite_noise(token):
    """
    Normalisation : Simplifie les n-grammes de lieux/orgs complexes en retirant  les mots génériques ou les particules.
    Ex: "Secteur de Mycogène" → "Secteur Mycogène"
    """
    parts = token.split()

    
    if len(parts) == 2 and parts[1].lower() == "monde":
        return parts[0]

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
    
    # Normaliser pour gérer les articles contractés
    normalized = normalize_articles(s).lower()

    if normalized in SEMANTIC_NOISE or s.lower() in SEMANTIC_NOISE:
        return True

    # Rejet des PER + bruit 
    if len(parts) >= 2 and RE_PROP.match(s.split()[0]) and parts[-1] in SEMANTIC_NOISE:
        return True

    if contains_verb_etre(s):
        return True
    
    if len(s) < 4:
        return True

    return False

def is_gentile_combo(s):
    """Vérifie les combinaisons Peuple + Lieu/Autre (ex: Terriens Spacetown)."""
    parts = s.split()
    # Si c'est un gentilé-lieu seul, le garder
    if len(parts) == 1 and s in GENTILES_LIEUX:
        return False
    # Rejeter les combinaisons comme "Terriens Spacetown"
    return len(parts) >= 2 and parts[0].lower() in GENTILES

def looks_like_location(token):
    
    t = token.strip()
    parts = t.split()

    
    if t in CANONICAL:
        return True
    
    if t in GENTILES_LIEUX:
        return True

    if not RE_PROP.match(parts[0]):
        return False

    if len(parts) == 1 and RE_PROP.match(t) and len(t) >= 4:
        if any(t.lower().endswith(s) for s in SUFFIXES):
            return True

    if len(parts)==2 and parts[0] in ("Secteur","Université","Palais") and parts[1] in CANONICAL:
        return True

    return False



def main():
    parser = argparse.ArgumentParser(description="Génère la liste LL par filtrage strict.")
    parser.add_argument("--L", type=Path, default="outputs/L.txt", help="Liste brute des candidats.")
    parser.add_argument("--LP", type=Path, default="outputs/LP_final.txt", help="Liste des personnages confirmés (pour l'exclusion).")
    parser.add_argument("-o","--output", type=Path, default="outputs/LL_final.txt", help="Fichier de sortie final (LL.txt).")
    args = parser.parse_args()

    L = load_list(args.L)
    LP = set(load_list(args.LP))  

    LL = set()  #

    print("--- Démarrage de la Tâche 3 (Extraction LL) ---")

    for item in L:

        # 1. Exclusion Personnages 
        if item in LP:
            continue

        # 2. Nettoyage et Normalisation
        base = clean_composite_noise(item)

        

        # 3. Rejet Bruit Sémantique 
        if is_semantic_garbage(base):
            continue

        # 4. Rejet Combinaison Peuple + Lieu
        if is_gentile_combo(base):
            continue

        # 5. Rejet Pronoms 
        if contains_pronoun(base):
            continue

    

        if looks_like_location(base):
            LL.add(base)

    out = sorted(list(LL))

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as f:
        for x in out:
            f.write(x+"\n")

    print(f"Terminé. {len(out)} lieux extraits.")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()