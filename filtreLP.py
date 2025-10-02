#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filtreLP.py
-----------
Extraction des PERSONNAGES probables (LP) à partir d'une liste brute L.

Entrées (dans l'ordre de priorité) :
  - L_counts.tsv : "candidat<TAB>compte" (recommandé)
  - L.txt        : une entrée par ligne (compte = 1)

Sorties :
  - LP.txt         : liste des personnages probables (une entrée par ligne)
  - LP_debug.tsv   : audit des décisions (candidat, count, decision, rule, score)

Heuristiques :
  - Accepte les séquences de 1 à 3 tokens capitalisés (InitCap ou ALLCAPS) en
    autorisant des particules basses ("de","du","d’") entre deux noms.
  - Favorise les patrons de type "Titre + Nom" (ex. "Monsieur Barr", "Dr Mis").
  - Exclut les mots fonctionnels (Il, Je, Le, Oui, Non, etc.), les chiffres romains,
    les ALLCAPS courts, les indices de lieux/organisations (Fondation, Planète, etc.).
  - Permet des exceptions via listes blanche/noire optionnelles.

Utilisation :
  python3 filtreLP.py
  python3 filtreLP.py --min-count 2 --out LP.txt --debug LP_debug.tsv
  python3 filtreLP.py --whitelist seeds_personnages.txt --blacklist a_exclure.txt
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import Iterable, Tuple, List, Dict

# ---------------------------------------------------------------------
# Jeux de données linguistiques (modulables)
# ---------------------------------------------------------------------

# Mots fonctionnels / formes dialogales fréquentes à exclure en tant que PERSONNES.
STOPWORDS_CAP = {
    "Je","Tu","Il","Elle","Nous","Vous","Ils","Elles",
    "Le","La","Les","L’","L'","Un","Une","Des","Du","De","D’","D'","Au","Aux",
    "Et","En","Que","Qu'","Qu’","Qui","Quoi","Quand","Pourquoi","Si",
    "Oui","Non","Merci","Oh","Où","N'", "N’", "Ne","Ni","Plus","Pas","Se",
    "Partie","Plans","Planète","Privé","Recherché"
}

# Début de séquences de type “phrase de dialogue” à ignorer comme PERSONNE.
LEAD_BAN = {
    "Oui","Non","Merci","Oh","Pourquoi","Quand","Quoi","Peut-être","Peutêtre",
    "Parfaitement","Patiemment","Prenez","Oubliez","Retournez","Réfléchis","Réveillez",
}

# Particules autorisées au milieu d’un nom composé.
LOWER_PARTICLES = {"de","du","des","d’","d'", "la","le","les"}

# Titres / civilités qui augmentent la probabilité d’un personnage.
TITLES = {
    "Monsieur","M.","Docteur","Dr","Gouverneur","Général","Comte","Duc",
    "Prince","Empereur","Majesté","Secrétaire"
}

# Indices forts que le candidat est plutôt un lieu/organisation que PERSONNE.
PLACE_HINTS = {
    "Port","Péninsule","Périphérie","Planète","Royaumes","Fondation","Empire",
}

# Quelques noms propres pertinents dans le cycle (facultatif, améliore rappel).
# Vous pouvez compléter ce set ou fournir un fichier --whitelist.
SOFT_WHITELIST = {
    "Mulet","Riose","Onum Barr","Ebling Mis","Toran","Bayta","Pritcher","Indbur",
    "Seldon","Magnifico","Randu","Siwenna","Néotrantor","Palley","Orum","Onum"
}

# Expressions régulières
ROMAN_RE = re.compile(r"^[IVXLCDM]{1,6}$")                  # chiffres romains courts
TOKEN_SPLIT_RE = re.compile(r"\s+")                         # découpe simple
APOST_RE = re.compile(r"[’']")                              # apostrophes
CAP_WORD_RE = re.compile(r"^[A-ZÉÈÊÀÂÎÏÔÛÇ][\w\-ÉÈÊÀÂÎÏÔÛÇéèêàâîïôûç]+$")


# ---------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------

def load_list(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def load_counts(path: Path) -> List[Tuple[str,int]]:
    items: List[Tuple[str,int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            cand, count = line.split("\t", 1)
        else:
            cand, count = line, "1"
        try:
            c = int(count.strip())
        except ValueError:
            c = 1
        items.append((cand.strip(), c))
    return items

def is_cap_token(tok: str) -> bool:
    """InitCap ou ALLCAPS (autorise diacritiques)."""
    return bool(tok) and (tok[0].isupper() or tok.isupper())

def is_allcaps_short(tok: str) -> bool:
    return tok.isupper() and 1 <= len(tok) <= 4

def is_roman_like(tok: str) -> bool:
    return bool(ROMAN_RE.match(tok))

def looks_like_dialog_phrase(toks: List[str]) -> bool:
    """Séquences dialogales typiques (ex: 'Oui Toran', 'Merci Ebling')."""
    if len(toks) >= 1 and toks[0] in LEAD_BAN:
        return True
    if len(toks) >= 2 and toks[1] in LEAD_BAN:
        return True
    return False

def contains_place_hint(toks: List[str]) -> bool:
    return any(t in PLACE_HINTS for t in toks)

def normalized_tokens(cand: str) -> List[str]:
    """Découpage sur espaces, conserve la casse (L est déjà 'propre')."""
    return TOKEN_SPLIT_RE.split(cand.strip())

def has_forbidden_apostrophe(tok: str) -> bool:
    """Exclut les formes fonctionnelles 'L'', 'D'', 'N'' isolées comme prénoms."""
    if len(tok) <= 2 and APOST_RE.search(tok):
        return True
    return False


# ---------------------------------------------------------------------
# Règles de décision PERSONNE
# ---------------------------------------------------------------------

def person_score(toks: List[str]) -> Tuple[int, str]:
    """
    Score heuristique et règle déclenchée (pour audit).
    Approche :
      - pénalités si motif roman-like, ALLCAPS court, stopwords en fin, head dialogal
      - bonus si patron Titre+Nom, 2-3 tokens capitalisés, particules basses admises
    """
    score = 0
    reason_parts: List[str] = []

    # Bruit dialogal
    if looks_like_dialog_phrase(toks):
        score -= 3
        reason_parts.append("dialog-lead")

    # Indices lieux/orga
    if contains_place_hint(toks):
        score -= 2
        reason_parts.append("place-hint")

    # Cas 1 token
    if len(toks) == 1:
        t = toks[0]
        if t in STOPWORDS_CAP:
            score -= 3; reason_parts.append("stopword")
        if is_allcaps_short(t):
            score -= 2; reason_parts.append("allcaps-short")
        if is_roman_like(t):
            score -= 3; reason_parts.append("roman-like")
        if has_forbidden_apostrophe(t):
            score -= 2; reason_parts.append("apostrophe-func")
        if is_cap_token(t) and t not in STOPWORDS_CAP and not is_allcaps_short(t) and not is_roman_like(t):
            score += 2; reason_parts.append("single-cap")
        return score, "+".join(reason_parts) if reason_parts else "single-eval"

    # Cas 2-3 tokens
    if 2 <= len(toks) <= 3:
        # Patron Titre + Nom (+ Nom)
        if toks[0] in TITLES and all(is_cap_token(x) for x in toks[1:]):
            score += 4; reason_parts.append("title+name")
        # Particules basses internes autorisées
        non_parts = [t for t in toks if t.lower() not in LOWER_PARTICLES]
        # Tous les non-particles doivent "ressembler" à des noms propres
        if all(is_cap_token(t) for t in non_parts):
            score += 3; reason_parts.append("cap-seq")
        # Pénalités
        if any(t in STOPWORDS_CAP for t in non_parts):
            score -= 2; reason_parts.append("stopword-in-seq")
        if any(is_allcaps_short(t) for t in non_parts):
            score -= 2; reason_parts.append("allcaps-short-in-seq")
        if any(is_roman_like(t) for t in non_parts):
            score -= 3; reason_parts.append("roman-in-seq")
        return score, "+".join(reason_parts) if reason_parts else "seq-eval"

    # Au-delà de 3 tokens : trop risqué sans contexte
    score -= 2
    reason_parts.append("too-long")
    return score, "+".join(reason_parts)


def is_person_candidate(cand: str, count: int,
                        whitelist: set[str],
                        blacklist: set[str],
                        min_count: int) -> Tuple[bool, int, str]:
    """
    Décision finale d'acceptation comme PERSONNE (True/False),
    accompagnée du score et de la règle principale déclenchée (pour audit).
    """
    if cand in blacklist:
        return False, -999, "blacklisted"
    if cand in whitelist or cand in SOFT_WHITELIST:
        return True, 999, "whitelisted"

    if count < min_count:
        return False, -100, "below-min-count"

    toks = normalized_tokens(cand)

    # Filtrages courts et évidents
    if any(tok in STOPWORDS_CAP for tok in toks):
        # si tout le candidat est un stopword ou majoritairement fonctionnel, rejeter
        if len(toks) == 1 or sum(t in STOPWORDS_CAP for t in toks) >= len(toks) - 1:
            return False, -10, "mostly-stopwords"

    # Score heuristique
    score, reason = person_score(toks)

    # Seuil de décision
    #   - Pour L très bruitée, un seuil >= 2 est souvent un bon compromis.
    #   - Ajustable via essais : baissez à 1 si rappel trop faible.
    accept = score >= 2
    return accept, score, reason


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def iter_candidates() -> Iterable[Tuple[str, int]]:
    """Source des candidats : L_counts.tsv prioritaire, sinon L.txt (count=1)."""
    lcounts = Path("L_counts.tsv")
    ltxt = Path("L.txt")

    if lcounts.exists():
        for cand, c in load_counts(lcounts):
            yield cand, c
        return

    if ltxt.exists():
        for cand in load_list(ltxt):
            yield cand, 1
        return

    raise SystemExit("Aucune source trouvée. Placez L_counts.tsv ou L.txt dans le répertoire courant.")


def main():
    ap = argparse.ArgumentParser(description="Filtrage de la liste L pour produire LP (personnages).")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Fréquence minimale pour considérer un candidat (défaut: 1).")
    ap.add_argument("--out", default="LP.txt",
                    help="Chemin du fichier de sortie LP.txt (défaut: LP.txt).")
    ap.add_argument("--debug", default="LP_debug.tsv",
                    help="Chemin du fichier d'audit LP_debug.tsv (défaut: LP_debug.tsv).")
    ap.add_argument("--whitelist", default=None,
                    help="Fichier texte (1 par ligne) de candidats garantis PERSONNE.")
    ap.add_argument("--blacklist", default=None,
                    help="Fichier texte (1 par ligne) de candidats à exclure.")
    args = ap.parse_args()

    whitelist = set(load_list(Path(args.whitelist))) if args.whitelist else set()
    blacklist = set(load_list(Path(args.blacklist))) if args.blacklist else set()

    accepted: List[str] = []
    debug_rows: List[Tuple[str,int,str,str,int]] = []  # (cand, count, decision, rule, score)

    for cand, cnt in iter_candidates():
        ok, score, rule = is_person_candidate(
            cand=cand,
            count=cnt,
            whitelist=whitelist,
            blacklist=blacklist,
            min_count=args.min_count
        )
        decision = "ACCEPT" if ok else "REJECT"
        debug_rows.append((cand, cnt, decision, rule, score))
        if ok:
            accepted.append(cand)

    # Déduplication en conservant l'ordre initial
    seen = set()
    lp_unique = []
    for c in accepted:
        if c not in seen:
            seen.add(c)
            lp_unique.append(c)

    # Écritures
    Path(args.out).write_text("\n".join(lp_unique), encoding="utf-8")

    with open(args.debug, "w", encoding="utf-8") as f:
        f.write("candidate\tcount\tdecision\trule\tscore\n")
        for cand, cnt, decision, rule, score in debug_rows:
            f.write(f"{cand}\t{cnt}\t{decision}\t{rule}\t{score}\n")

    print(f"LP écrit : {args.out} ({len(lp_unique)} entrées)")
    print(f"Audit écrit : {args.debug}")

if __name__ == "__main__":
    main()
