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
TOKEN_RE = re.compile(rf"[{_LET}]+(?:['’][{_LET}]+|-[{_LET}]+)*")

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
    """Découpe le texte en tokens."""
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
    Construit une séquence "gourmande" valide (PN + particule + PN...).
    S'arrête si la chaîne est rompue.
    """
    n = len(tokens)
    if not cap_mask[start_index]:
        return None

    seq = [tokens[start_index]]
    j = start_index + 1
    expect_particle = True # Après un nom, on peut avoir une particule

    while j < n and len(seq) < max_len:
        token = tokens[j]
        
        if expect_particle:
            if token.lower() in PARTICLES:
                seq.append(token)
                expect_particle = False # Après particule, on veut obligatoirement un Nom
                j += 1
                continue
            # Si pas de particule, on regarde si c'est un autre Nom (ex: "Hari Seldon")
            # On sort de la boucle pour traiter le cas "Nom Nom" ci-dessous
        
        if cap_mask[j]:
            seq.append(token)
            expect_particle = True # Après un Nom, on peut avoir une particule
            j += 1
        else:
            break # Ni particule, ni Nom -> Fin de séquence

    # Validation finale : La séquence ne doit pas finir par une particule
    while seq and seq[-1].lower() in PARTICLES:
        seq.pop()

    if len(seq) >= 1:
        return " ".join(seq)
    return None

def generate_candidates(text: str, antidico: Set[str], use_spacy_filter: bool = True) -> Counter:
    """
    Fonction principale qui génère les candidats.
    """
    # 1. Tokenisation
    tokens = tokenize(text)
    cap_mask = [t and t[0].isupper() for t in tokens]
    sentence_starts = get_sentence_starts(text)
    counts = Counter()
    
    # 2. Construction de la stop-list dynamique (optionnel mais recommandé)
    dynamic_stops = set()
    if use_spacy_filter:
        dynamic_stops = build_dynamic_stoplist(text)
    
    full_antidico = antidico | dynamic_stops | CONVERSATIONAL_STARTERS

    # 3. Balayage du texte
    n = len(tokens)
    for i in range(n):
        token = tokens[i]
        
        # Si le mot n'est pas capitalisé, on passe
        if not cap_mask[i]:
            continue
            
        # --- FILTRES ---
        # 1. Si c'est un début de phrase et que le mot est commun -> Rejet
        if i in sentence_starts and token.lower() in full_antidico:
            continue
        
        # 2. Si c'est un mot conversationnel ou un chiffre romain -> Rejet
        if token in CONVERSATIONAL_STARTERS or token in ROMAN_NUMERALS:
            continue
            
        # 3. Filtre anti-OCR (lettres répétées comme "JJ")
        if len(token) == 2 and token[0] == token[1]:
            continue

        # 4. Filtre antidictionnaire strict (même hors début de phrase)
        if token.lower() in full_antidico:
            continue

        # --- GÉNÉRATION ---
        # On essaie de construire le n-gramme le plus long possible
        phrase = build_ngram_greedy(tokens, cap_mask, i)
        
        if phrase:
            # On ne garde que les candidats d'une longueur raisonnable (> 2 lettres)
            if len(phrase) > 2:
                counts[phrase] += 1

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