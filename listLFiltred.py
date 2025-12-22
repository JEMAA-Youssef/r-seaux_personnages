#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
listLFiltred.py — Génération avancée de la liste des candidats (L)

Ce script réalise la Tâche 1 du projet : produire une liste d'entités nommées candidates.
Il utilise une approche hybride combinant des règles linguistiques (n-grammes, capitalisation)
et des outils NLP modernes (SpaCy) pour un filtrage intelligent dès la source.

Fonctionnalités clés :
1. Normalisation robuste du texte (Unicode, apostrophes).
2. Génération de n-grammes "gourmands" (PN + particule + PN).
3. Filtrage dynamique via antidictionnaire et stop-list grammaticale (verbes, adverbes).
4. Étiquetage final des candidats (PER, LOC, ORG) via SpaCy NER.
5. Export en formats .txt (simple) et .tsv (enrichi).

Auteur : [Votre Nom / Binôme]
Date   : Septembre 2025
"""

import argparse
import unicodedata
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple

# =============================================================================
# CONFIGURATION & CONSTANTES
# =============================================================================

# Regex pour identifier les tokens (mots avec lettres, apostrophes, tirets)
_LET = r"A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"
# Regex pour un MOT (identique à votre version précédente)
WORD_PATTERN = rf"[{_LET}]+(?:['’`\u2019][{_LET}]+|-[{_LET}]+)*"

# Regex pour la PONCTUATION BLOQUANTE (virgules, points, etc.)
# Ce sont les murs que l'algorithme ne doit pas traverser.
PUNCT_PATTERN = r"[.,;?!:()«»“”\"]"

# Tokenizer combiné : On capture soit un Mot, soit une Ponctuation
# Le (?:...) signifie "groupe non capturant" pour garder une liste plate
TOKEN_RE = re.compile(rf"(?:{WORD_PATTERN})|(?:{PUNCT_PATTERN})")

# Regex pour repérer les fins de phrases (pour le contexte grammatical)
SENT_BOUNDARY_RE = re.compile(r'(?<=[\.\!\?\;\:\u2026])\s+|[\n]+')

# Particules autorisées dans les noms composés (ex: "de", "la")
PARTICLES = {
    "de", "du", "des", "d'", "d’", "l'", "l’", "le", "la", "les",
    "à", "au", "aux", "en", "pour", "sur", "van", "von"
}

# Mots qui commencent souvent une phrase mais ne sont pas des noms
CONVERSATIONAL_STARTERS = {
    "Allez", "Allons", "Alors", "Asseyez-vous", "Assez", "Attendez",
    "Aussi", "Bon", "Bonjour", "Bonsoir", "Cependant", "Comment", 
    "Continuez", "Dis", "Dites", "Ecoute", "Ecoutez", "Entendu", "Et", 
    "Impossible", "Mais", "Merci", "Non", "Oh", "Oui", "Or", "Pauvre", 
    "Pourquoi", "Puis", "Quand", "Quoi", "Regardez", "Seul", "Voyons", 
    "C'est", "C’est", "J'ai", "C'était", "Qu'est-ce", "Est-ce"
}

# Chiffres romains à filtrer (souvent des numéros de chapitre)
ROMAN_NUMERALS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}

# =============================================================================
# FONCTIONS UTILITAIRES (TEXTE & FICHIERS)
# =============================================================================

def load_antidictionary(file_path: Path) -> Set[str]:
    """Charge l'antidictionnaire depuis un fichier texte (un mot par ligne)."""
    if not file_path.is_file():
        print(f"Attention : Antidictionnaire introuvable à {file_path}")
        return set()
    with file_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

def normalize_text(text: str) -> str:
    """
    Nettoie et normalise le texte brut.
    - Normalisation Unicode (NFKC)
    - Unification des apostrophes et guillemets
    - Suppression des césures de fin de ligne (mot-\n suite)
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("`", "'").replace("´", "'")
    t = t.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    t = re.sub(r"-\s*\n\s*", "", t) # Fusionne les mots coupés
    t = t.replace("\u00A0", " ").replace("\t", " ")
    t = re.sub(r"[ ]{2,}", " ", t) # Réduit les espaces multiples
    return t

def tokenize(text: str) -> List[str]:
    """
    Découpe le texte en tokens (mots ET ponctuation).
    Les espaces sont ignorés.
    """
    # findall avec notre nouvelle regex va récupérer les mots et la ponctuation
    return TOKEN_RE.findall(text)

def get_sentence_starts(text: str) -> Set[int]:
    """
    Identifie les indices des tokens qui commencent une phrase.
    Utile pour filtrer les mots communs capitalisés en début de phrase.
    """
    starts = set()
    token_index = 0
    # Découpage basique par phrase
    for segment in re.split(SENT_BOUNDARY_RE, text):
        if not segment.strip():
            continue
        # Nettoyage des marques de dialogue au début
        segment = segment.lstrip('«»"“”\'—–- ').lstrip()
        seg_tokens = tokenize(segment)
        if seg_tokens:
            starts.add(token_index)
            token_index += len(seg_tokens)
    return starts

# =============================================================================
# LOGIQUE LINGUISTIQUE & SPACY
# =============================================================================

def build_dynamic_stoplist(text: str, model: str = "fr_core_news_md") -> Set[str]:
    """
    Utilise SpaCy pour construire une liste dynamique de mots à exclure (verbes, adverbes).
    Ceci complète l'antidictionnaire statique.
    """
    try:
        import spacy
        nlp = spacy.load(model, disable=["ner", "parser"]) # On a juste besoin du tagger
    except ImportError:
        print("Erreur : La librairie 'spacy' n'est pas installée.")
        sys.exit(1)
    except OSError:
        print(f"Erreur : Le modèle '{model}' n'est pas trouvé. Installez-le avec : python -m spacy download {model}")
        sys.exit(1)

    print("Analyse grammaticale du corpus pour filtrage dynamique...")
    # Augmentation de la limite de taille pour les gros corpus
    nlp.max_length = max(len(text) + 100000, 1500000)
    
    doc = nlp(text[:1000000]) # Analyse le premier million de caractères pour construire la liste
    stops = set()

    for token in doc:
        # Exclure les verbes, auxiliaires, adverbes, pronoms, déterminants
        if token.pos_ in {"VERB", "AUX", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "INTJ"}:
            stops.add(token.text.lower())
            stops.add(token.lemma_.lower())
    
    return stops

def tag_candidates(counts: Counter, model: str = "fr_core_news_md") -> List[Tuple[str, int, str]]:
    """
    Associe une étiquette (PER, LOC, ORG, UNK) à chaque candidat généré.
    Utilise le modèle NER de SpaCy.
    """
    try:
        import spacy
        nlp = spacy.load(model) # Charge le modèle complet avec NER
    except Exception:
        return [(k, v, "UNK") for k, v in counts.items()]

    tagged_results = []
    candidates = list(counts.keys())
    
    print(f"Étiquetage final de {len(candidates)} candidats...")
    
    # Traitement par lots pour la performance
    for doc in nlp.pipe(candidates, batch_size=50):
        text = doc.text
        freq = counts[text]
        label = "UNK"

        if doc.ents:
            # Récupère tous les labels trouvés dans l'expression
            labels = [ent.label_ for ent in doc.ents]
            # Priorité hiérarchique pour notre projet
            if "PER" in labels:
                label = "PER"
            elif "LOC" in labels:
                label = "LOC"
            elif "ORG" in labels:
                label = "ORG"
            elif "MISC" in labels:
                label = "MISC"
            else:
                label = labels[0]
        
        tagged_results.append((text, freq, label))

    # Tri : D'abord les PER, puis par fréquence décroissante
    tagged_results.sort(key=lambda x: (x[2] != "PER", -x[1]))
    return tagged_results

# =============================================================================
# CŒUR DE L'ALGORITHME : CONSTRUCTION DES CANDIDATS
# =============================================================================

def build_ngram_greedy(tokens: List[str], cap_mask: List[bool], start_index: int, max_len: int = 6) -> str | None:
    """
    Construit une séquence valide en s'arrêtant net à la moindre ponctuation.
    """
    n = len(tokens)
    
    # Si le mot de départ n'est pas une majuscule (ou est une ponctuation), on annule
    if not cap_mask[start_index]: 
        return None

    # Initialisation
    seq = [tokens[start_index]]
    j = start_index + 1
    expect_particle = True 

    while j < n and len(seq) < max_len:
        token = tokens[j]
        
        # --- NOUVEAU : Le Mur de Ponctuation ---
        # Si le token est une ponctuation (virgule, point...), on arrête TOUT de suite.
        # On vérifie ça simplement : ce n'est ni un mot capitalisé, ni une particule, ni un mot minuscule accepté.
        # Une virgule n'est jamais dans PARTICLES et n'est jamais isupper().
        
        if token in PARTICLES:
            if expect_particle:
                seq.append(token)
                expect_particle = False # Après une particule ("de"), on VEUT un Nom
                j += 1
                continue
            else:
                # On a "Nom Particule Particule" ? Non, on arrête.
                break
        
        # Si c'est un mot avec Majuscule (Nom Propre)
        if cap_mask[j]:
            seq.append(token)
            expect_particle = True # Après un Nom, on peut avoir une particule ou un autre Nom
            j += 1
            continue
            
        # Si on arrive ici, ce n'est ni une particule, ni une majuscule.
        # C'est donc soit un mot minuscule ordinaire, soit une VIRGULE/POINT.
        # Dans les deux cas -> FIN DE LA SÉQUENCE.
        break

    # Validation finale
    
    # 1. On ne finit pas par une particule (ex: "Duc de")
    while seq and seq[-1] in PARTICLES:
        seq.pop()

    # 2. Nettoyage des ponctuations qui auraient pu se glisser (par sécurité)
    # Bien que notre logique devrait les avoir empêchés d'entrer.
    if not seq:
        return None

    # On renvoie la séquence seulement si elle est valide
    return " ".join(seq)

def generate_candidates(text: str, antidico: Set[str], use_spacy_filter: bool = True) -> Counter:
    """
    Fonction principale qui génère les candidats.
    Intègre tous les filtres originaux + gestion de la ponctuation.
    """
    # 1. Tokenisation (inclut maintenant la ponctuation)
    tokens = tokenize(text)
    
    # Masque : True si Majuscule, False si minuscule OU ponctuation
    cap_mask = [t[0].isupper() for t in tokens]
    
    # Note : get_sentence_starts utilise tokenize(), donc il restera synchronisé
    sentence_starts = get_sentence_starts(text) 
    
    counts = Counter()
    
    # 2. Stop-list
    dynamic_stops = set()
    if use_spacy_filter:
        dynamic_stops = build_dynamic_stoplist(text)
    
    full_antidico = antidico | dynamic_stops | CONVERSATIONAL_STARTERS

    # 3. Balayage
    n = len(tokens)
    i = 0
    while i < n:
        token = tokens[i]
        
        # Optimisation : Si c'est une ponctuation ou une minuscule, on passe
        if not cap_mask[i]:
            i += 1
            continue
            
        # --- VOS FILTRES (Regroupés pour clarté) ---
        
        # A. Filtre début de phrase + mot commun
        is_start_common = (i in sentence_starts and token.lower() in full_antidico)
        
        # B. Filtre Conversationnel / Chiffres Romains / Antidico Strict
        is_forbidden = (token in CONVERSATIONAL_STARTERS) or \
                       (token in ROMAN_NUMERALS) or \
                       (token.lower() in full_antidico)
                       
        if is_start_common or is_forbidden:
            i += 1
            continue

        # C. Filtre Anti-OCR (Lettres répétées ex: "JJ")
        if len(token) == 2 and token[0] == token[1]:
            i += 1
            continue

        # --- GÉNÉRATION ---
        phrase = build_ngram_greedy(tokens, cap_mask, i)
        
        if phrase:
            if len(phrase) > 2:
                counts[phrase] += 1
            
            # Ici, on ne saute pas d'index pour l'instant (i += 1) pour être sûr 
            # de ne rien rater, conformément à votre logique originale.
            
        i += 1

    return counts

# =============================================================================
# FONCTION PRINCIPALE (MAIN)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tâche 1 : Génération de la liste de candidats (L)")
    parser.add_argument("corpus_files", nargs="+", help="Fichiers .txt.preprocessed à analyser")
    parser.add_argument("-o", "--outdir", default="outputs", help="Dossier de sortie (défaut: outputs)")
    parser.add_argument("--antidico", default="resources/antidictionnaire.txt", help="Chemin vers l'antidictionnaire")
    parser.add_argument("--spacy-model", default="fr_core_news_md", help="Modèle SpaCy (défaut: fr_core_news_md)")
    args = parser.parse_args()

    # 1. Chargement des ressources
    print(f"--- Démarrage de la Tâche 1 ---")
    print(f"Chargement de l'antidictionnaire : {args.antidico}")
    antidico = load_antidictionary(Path(args.antidico))

    # 2. Lecture du corpus
    print(f"Lecture de {len(args.corpus_files)} fichiers corpus...")
    full_text = ""
    for fpath in args.corpus_files:
        try:
            text = Path(fpath).read_text(encoding="utf-8")
            full_text += text + "\n"
        except Exception as e:
            print(f"Erreur lecture {fpath}: {e}")
    
    # 3. Normalisation
    print("Normalisation du texte...")
    normalized_text = normalize_text(full_text)

    # 4. Génération des candidats (Cœur du travail)
    print("Génération des candidats (avec filtrage dynamique)...")
    counts = generate_candidates(normalized_text, antidico, use_spacy_filter=True)
    print(f"-> {len(counts)} candidats uniques trouvés.")

    # 5. Étiquetage final (Pour Tâche 2)
    print("Étiquetage NER des candidats...")
    tagged_items = tag_candidates(counts, model=args.spacy_model)

    # 6. Écriture des résultats
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fichier L.txt (Liste simple, exigence académique)
    l_txt = out_dir / "L.txt"
    with l_txt.open("w", encoding="utf-8") as f:
        for txt, _, _ in tagged_items:
            f.write(f"{txt}\n")
    
    # Fichier L_tagged.tsv (Liste enrichie, pour l'étape suivante)
    l_tsv = out_dir / "L_tagged.tsv"
    with l_tsv.open("w", encoding="utf-8") as f:
        f.write("Candidat\tFrequence\tLabel\n")
        for txt, freq, label in tagged_items:
            f.write(f"{txt}\t{freq}\t{label}\n")

    print(f"\n--- Terminé avec succès ---")
    print(f"Fichiers générés :")
    print(f"  1. {l_txt} (Liste brute nettoyée)")
    print(f"  2. {l_tsv} (Liste enrichie avec tags et fréquences)")

if __name__ == "__main__":
    main()


#python3 listLFiltred.py corpus_asimov_leaderboard/*/*.txt.preprocessed -o outputs