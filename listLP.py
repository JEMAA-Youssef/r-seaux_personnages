#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import unicodedata
import re
from collections import Counter
from pathlib import Path

# --- Constants ---
_LET = r"A-Za-zÀ-ÖØ-öø-ÿŒœÆæ"
TOKEN_RE = re.compile(rf"[{_LET}]+(?:['’][{_LET}]+|-[{_LET}]+)*")
PARTICLES = {
    "de", "du", "des", "d'", "d’", "l'", "l’", "le", "la", "les",
    "à", "au", "aux", "en", "pour", "sur"
}
CONVERSATIONAL_STARTERS = {
    "Allez", "Allons", "Alors", "Asseyez-vous", "Assez", "Assieds-toi", "Attendez",
    "Aussi", "Bon", "Bonjour", "Bonsoir", "Cependant", "Comment", "Continuez", "Dis", "Dites",
    "Ecoute", "Ecoutez", "Entendu", "Et", "Impossible", "Mais", "Merci", "Non", "Oh", "Oui",
    "Or", "Pauvre", "Pourquoi", "Puis", "Quand", "Quoi", "Regardez", "Seul", "Voyons", "C'est", "C’est",
    "J'ai", "C'était", "Qu'est-ce", "Est-ce", "J'en", "J'avais", 
    "J'espère", "J'aurais", "Voulez-vous", "Pouvez-vous", "Avez-vous",
    "Pensez-vous", "Savez-vous", "Croyez-vous", "Venez", "Restez", "Tenez"
}

#un filtre pour les chiffres romains
ROMAN_NUMERALS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}

# un filtre pour les articles contractés
def is_contracted_article(token: str) -> bool:
    return token.startswith(("L'", "D'", "N'", "S'", "J'", "C'", "Qu'", "Jusqu'", "Aujourd'"))


# Repérage des frontières de phrase (., !, ?, ;, :, …) + retours ligne
SENT_BOUNDARY_RE = re.compile(r'(?<=[\.\!\?\;\:\u2026])\s+|[\n]+')


# --- Helper Functions ---

def spacy_dynamic_antidico(text: str, model: str = "fr_core_news_md") -> set[str]:
    """
    Construit dynamiquement un antidictionnaire basé sur l'analyse spaCy.
    
    Filtre automatiquement :
    - Verbes, auxiliaires, interjections, impératifs
    - Adverbes et adjectifs en début de phrase
    - Formes contractées (Aujourd', Jusqu', etc.)
    - Verbes dans les questions
    - Mots fonctionnels et grammaticaux
    - Formes VERB-PRON hyphénées (Faites-le, Dites-moi, etc.)
    
    Args:
        text: Texte à analyser
        model: Modèle spaCy à utiliser
        
    Returns:
        Set des mots supplémentaires à exclure
    """
    import spacy
    
    try:
        # Charger le modèle avec tous les composants nécessaires
        nlp = spacy.load(model, disable=["ner"])
    except OSError:
        raise ImportError(f"Modèle spaCy '{model}' non trouvé. Installez-le avec: python -m spacy download {model}")
    
    doc = nlp(text)
    stops: set[str] = set()
    
    # Détecter les phrases pour identifier les débuts de phrase
    sentences = list(doc.sents)
    sentence_starts = {sent[0].i for sent in sentences}
    
    # Détecter les phrases interrogatives
    question_sentences = set()
    for sent in sentences:
        if sent.text.strip().endswith('?'):
            question_sentences.update(tok.i for tok in sent)
    
    # Clitiques fréquents pour détecter les formes VERB-PRON
    clitics = {"moi", "toi", "lui", "nous", "vous", "leur", "le", "la", "les", "y", "en", "l", "t"}
    
    for tok in doc:
        # Ignorer les tokens vides ou trop courts
        if not tok.text.strip() or len(tok.text.strip()) < 2:
            continue
            
        # Ignorer les noms propres (PROPN) - on veut les garder !
        if tok.pos_ == "PROPN":
            continue
            
        # 1. Verbes, auxiliaires, interjections et impératifs
        if tok.pos_ in {"VERB", "AUX", "INTJ"} or "Imp" in tok.morph.get("Mood", []):
            _add_token_variants(stops, tok)
            continue
            
        # 1bis. Détection des impératifs par morphologie (plus robuste)
        if tok.pos_ == "VERB" and tok.morph.get("Mood") == ["Imp"]:
            _add_token_variants(stops, tok)
            continue
            
        # 2. Adverbes et adjectifs en début de phrase
        if tok.pos_ in {"ADV", "ADJ"} and tok.i in sentence_starts:
            _add_token_variants(stops, tok)
            continue
            
        # 3. Formes contractées (contractions avec apostrophe)
        if "'" in tok.text or "'" in tok.text:
            # Vérifier si c'est une contraction fonctionnelle
            if _is_functional_contraction(tok):
                _add_token_variants(stops, tok)
                continue
                
        # 4. Verbes dans les questions
        if tok.i in question_sentences and tok.pos_ in {"VERB", "AUX"}:
            _add_token_variants(stops, tok)
            continue
            
        # 5. Mots fonctionnels et grammaticaux
        if tok.pos_ in {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "PART"}:
            _add_token_variants(stops, tok)
            continue
            
        # 6. Adverbes et adjectifs courants (même hors début de phrase)
        if tok.pos_ in {"ADV", "ADJ"} and _is_common_adverb_adjective(tok):
            _add_token_variants(stops, tok)
            continue
    
    # 7. Détecter les formes VERB-PRON hyphénées dans le texte
    for sent in doc.sents:
        tokens = list(sent)
        for i in range(len(tokens) - 1):
            current_tok = tokens[i]
            next_tok = tokens[i + 1]
            
            # Si VERB/AUX suivi d'un PRON clitique
            if (current_tok.pos_ in {"VERB", "AUX"} and 
                next_tok.pos_ == "PRON" and 
                (next_tok.lemma_.lower() in clitics or next_tok.text.lower().strip("'") in clitics)):
                
                # Créer la forme hyphénée
                hyphen_form = f"{current_tok.text}-{next_tok.text}"
                stops.add(hyphen_form)
                stops.add(hyphen_form.capitalize())
                stops.add(hyphen_form.lower())
    
    # 8. Détecter les formes VERB-PRON déjà hyphénées (un seul token)
    for tok in doc:
        if "-" in tok.text and tok.pos_ in {"VERB", "AUX"}:
            left, right = tok.text.split("-", 1)
            if right.lower().strip("'") in clitics:
                # Ajouter toutes les variantes
                stops.add(tok.text)
                stops.add(tok.text.lower())
                stops.add(tok.text.capitalize())
    
    # 9. Détecter les questions et impératifs spécifiques
    question_imperative_patterns = {
        "as-tu", "peux-tu", "veux-tu", "dois-tu", "peut-il", "veut-il",
        "fais-le", "dis-lui", "dites-lui", "regarde-moi", "écoute-moi",
        "tais-toi", "reste-là", "va-t-en", "viens-ici", "reste-ici",
        # Impératifs spécifiques trouvés dans vos résultats
        "entrez", "epargnez-moi", "epargnons", "essayons", "expliquez-moi",
        "finissons-en", "flairez-vous", "précédez-nous", "retenez-le", "sachez",
        "dormez", "tais-toi", "restez", "venez", "allez", "prenez", "donnez",
        "montrez", "racontez", "écoutez", "regardez", "faites", "dites"
    }
    
    for pattern in question_imperative_patterns:
        stops.add(pattern)
        stops.add(pattern.capitalize())
        stops.add(pattern.upper())
    
    # Ajouter des cas spéciaux fréquents
    _add_common_stopwords(stops)
    
    return stops


def _add_token_variants(stops: set[str], tok) -> None:
    """Ajoute différentes variantes d'un token à l'antidictionnaire."""
    if not tok.text.strip():
        return
        
    # Forme originale
    stops.add(tok.text)
    stops.add(tok.text.lower())
    stops.add(tok.text.capitalize())
    
    # Lemme si différent
    if tok.lemma_ and tok.lemma_ != tok.text:
        stops.add(tok.lemma_)
        stops.add(tok.lemma_.lower())
        stops.add(tok.lemma_.capitalize())


def _is_functional_contraction(tok) -> bool:
    """Détermine si une contraction est fonctionnelle (à exclure)."""
    text = tok.text.lower()
    
    # Contractions fonctionnelles communes
    functional_patterns = [
        "aujourd'", "jusqu'", "lorsqu'", "puisqu'", "quoiqu'",
        "quelqu'", "quelques", "quelquefois", "quelque",
        "presqu'", "entr'", "jusqu'à", "jusqu'alors"
    ]
    
    for pattern in functional_patterns:
        if text.startswith(pattern):
            return True
            
    return False


def _is_common_adverb_adjective(tok) -> bool:
    """Détermine si un adverbe/adjectif est commun (à exclure)."""
    text = tok.text.lower()
    
    # Adverbes et adjectifs courants à exclure
    common_words = {
        "exactement", "absolument", "vraiment", "naturellement", 
        "lentement", "rapidement", "facilement", "difficilement",
        "probablement", "certainement", "évidemment", "apparemment",
        "exact", "vrai", "faux", "bon", "mauvais", "grand", "petit",
        "nouveau", "ancien", "jeune", "vieux", "beau", "laid",
        "intéressant", "ennuyeux", "drôle", "sérieux", "important",
        "facile", "difficile", "possible", "impossible", "nécessaire"
    }
    
    return text in common_words


def _add_common_stopwords(stops: set[str]) -> None:
    """Ajoute des mots fonctionnels courants à l'antidictionnaire."""
    common_stops = {
        # Questions et impératifs
        "faites", "dites", "regardez", "écoutez", "venez", "allez",
        "prenez", "donnez", "montrez", "expliquez", "racontez",
        "êtes-vous", "pouvez-vous", "voulez-vous", "avez-vous",
        "pouvons-nous", "voulons-nous", "allons-nous","nous-mêmes","supposez","asseyez","mangeons","réfléchis",
        "périphérie",
        
        # Formes VERB-PRON spécifiques
        "faites-le", "dis-lui", "dites-lui", "dormez", "as-tu", "peux-tu", "tais-toi",
        "regarde-moi", "écoute-moi", "reste-là", "va-t-en", "viens-ici",
        
        # Impératifs spécifiques trouvés dans les résultats
        "entrez", "epargnez-moi", "epargnons", "essayons", "expliquez-moi",
        "finissons-en", "flairez-vous", "précédez-nous", "retenez-le", "sachez","imaginez-vous","imaginezvous",
        
        # Adjectifs/adverbes isolés
        "désolé", "démocrate", "intérieure", "extérieure","conseil",
        
        # Adverbes temporels et modaux
        "maintenant", "aujourd'hui", "hier", "demain", "bientôt",
        "toujours", "jamais", "souvent", "rarement", "parfois",
        "peut-être", "probablement", "certainement", "sûrement",
        "marchand","maire","marchands","excellence","seigneur","magnifiée","majesté","monsieur","attention",
        "capitaine",
        
        # Interjections et exclamations
        "ah", "oh", "eh", "hé", "holà", "tiens", "voilà", "voici",
        "bon", "bien", "mal", "tant", "assez", "trop", "très","aïnsi",
        
        # Mots de liaison
        "donc", "alors", "ainsi", "cependant", "néanmoins", "pourtant",
        "toutefois", "néanmoins", "cependant", "mais", "et", "ou","etre",
        
        # Formes contractées
        "aujourd'hui", "jusqu'à", "jusqu'alors", "lorsqu'il", "puisqu'il","lorsqu'ils",
        "quoiqu'il", "quelqu'un", "quelques", "quelque", "presque","jour","jours","partie",

        #Chiffres
        "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
        "vingt", "trente", "quarante", "cinquante", "soixante", "soixante-dix", "quatre-vingts", "quatre-vingt-dix",
        "cent", "deux-cents", "trois-cents", "quatre-cents", "cinq-cents", "six-cents", "sept-cents", "huit-cents", "neuf-cents",
        "deuxième","vingtième"
    }
    
    for word in common_stops:
        stops.add(word)
        stops.add(word.capitalize())
        stops.add(word.upper())

def postfilter_pos(counts: Counter, nlp) -> Counter:
    """
    Post-filtre les candidats finaux par POS pour éliminer les derniers faux positifs.
    
    Retire :
    - unigrams ADJ/ADV
    - unigrams VERB au participe passé (VerbForm=Part)
    - formes VERB-PRON résiduelles
    - petits sigles tout-majuscules de 1–2 lettres (bruit OCR) hors whitelist
    
    Args:
        counts: Counter des candidats avec leurs fréquences
        nlp: Modèle spaCy pour l'analyse POS
        
    Returns:
        Counter filtré
    """
    if nlp is None:
        return counts
        
    filtered = Counter()
    for expr, c in counts.items():
        doc = nlp(expr)
        
        # cas unigramme
        if len(doc) == 1:
            t = doc[0]
            if t.pos_ in {"ADJ", "ADV"}:
                continue
            if t.pos_ == "VERB" and "Part" in t.morph.get("VerbForm", []):
                continue
            if len(t.text) <= 2 and t.text.isupper() and t.lemma_.lower() not in {"iv", "vi", "ix"}:
                continue

        # filtre VERB-PRON si spaCy les re-segmente en 2 tokens
        if "-" in expr:
            parts = expr.split("-", 1)
            d = nlp(" ".join(parts))
            if len(d) == 2 and d[0].pos_ in {"VERB", "AUX"} and d[1].pos_ == "PRON":
                continue

        filtered[expr] = c
    return filtered

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

def is_stop(tok: str, antidico: set[str]) -> bool:
    return tok.lower() in antidico



def is_capitalized(tok: str) -> bool:
    return tok and tok[0].isupper()

def sentence_starts(text: str, token_re: re.Pattern = TOKEN_RE) -> set[int]:
    """
    Renvoie l'ensemble des indices de tokens qui sont les 1ers tokens de phrase.
    La méthode : split du texte en segments de phrase, tokenise chaque segment,
    et marque le premier token de chaque segment comme 'début de phrase'.
    """
    starts = set()
    token_index = 0
    # on split : chaque segment ≈ une phrase (simple mais efficace pour notre usage)
    for segment in re.split(SENT_BOUNDARY_RE, text):
        if not segment.strip():
            continue
        # enlève les guillemets ou tirets au début du segment (cas des dialogues)
        segment = segment.lstrip('«»"“”\'—–- ').lstrip()

        seg_tokens = token_re.findall(segment)
        if seg_tokens:
            starts.add(token_index)      # le 1er token de ce segment est en début de phrase
            token_index += len(seg_tokens)
    return starts




def load_antidictionary(file_path: Path) -> set[str]:
    """Loads the antidictionary into a set for fast lookups."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Antidictionary not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}
    
def normalize(text: str) -> str:
    """
    Nettoie et homogénéise le texte pour éviter les erreurs de tokenisation :
    - Normalisation Unicode (NFKC)
    - Homogénéisation des apostrophes
    - Suppression des césures (mot-\n suite)
    - Suppression des espaces spéciaux
    """
    # Normalisation canonique Unicode (transforme les caractères équivalents)
    t = unicodedata.normalize("NFKC", text)

    # Uniformiser les apostrophes et guillemets
    t = t.replace("’", "'").replace("`", "'").replace("´", "'")
    t = t.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')

    # Supprimer les traits d’union en fin de ligne (souvent césures OCR)
    t = re.sub(r"-\s*\n\s*", "", t)

    # Remplacer les espaces insécables et autres blancs Unicode par des espaces simples
    t = t.replace("\u00A0", " ").replace("\u202F", " ").replace("\t", " ")

    # Réduire les doubles espaces
    t = re.sub(r"[ ]{2,}", " ", t)

    return t



def build_ngram_greedy(
    tokens: list[str],
    cap: list[bool],
    i: int,
    min_len: int = 3,
    max_len: int = 8,
    require_particle: bool = False,
) -> str | None:
    """
    Construit une séquence de type: PN (particule PN)+ à partir du token i.

    Règles:
    - commence sur un token capitalisé (cap[i] == True)
    - enchaîne: particule -> PN -> particule -> PN ...
    - s'arrête dès qu'un maillon manque
    - doit se terminer sur un PN (pas sur une particule)
    - longueur >= min_len
    - si require_particle=True, au moins une particule doit être utilisée
    """
    n = len(tokens)
    if not cap[i]:
        return None

    seq = [tokens[i]]
    j = i + 1
    expect_particle = True   # après un PN, on attend une particule
    used_particle = False

    while j < n and len(seq) < max_len:
        tj = tokens[j]
        if expect_particle:
            # on veut une particule entre deux PN successifs
            if tj.lower() in PARTICLES:
                seq.append(tj)
                used_particle = True
                expect_particle = False  # prochain attendu: PN
                j += 1
                continue
            break
        else:
            # on veut un PN (token capitalisé)
            if cap[j]:
                seq.append(tokens[j])
                expect_particle = True   # prochain attendu: particule
                j += 1
                continue
            break

    # Conditions de validité:
    # - terminé sur un PN => expect_particle doit être True
    # - longueur minimale
    # - si require_particle: au moins une particule utilisée
    if expect_particle and len(seq) >= min_len and (used_particle or not require_particle):
        return " ".join(seq)
    return None


def _as_words(s: str) -> list[str]:
    return s.split()

def _is_token_subspan(longer: str, shorter: str) -> bool:
    """
    True si `shorter` est une sous-séquence CONTIGUË de `longer` au niveau tokens.
    Evite les faux matches par simple .find() sur chaîne.
    """
    lw, sw = _as_words(longer), _as_words(shorter)
    if len(sw) >= len(lw):
        return False
    n, m = len(lw), len(sw)
    for i in range(0, n - m + 1):
        if lw[i:i+m] == sw:
            return True
    return False

def prune_substrings(counts: Counter, min_gain: int = 1) -> Counter:
    """
    Supprime les n-grammes 'shorter' qui sont strictement inclus (en tant que sous-span de tokens)
    dans une forme plus longue 'longer' dont la fréquence est >= (freq(shorter) - min_gain).
    Logique: si la courte n'apporte pas de 'gain' de fréquence, on garde la longue.
    """
    # On travaille sur une liste triée pour favoriser les plus fortes et les plus longues
    items = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    kept = []  # liste de tuples (expr, freq)
    for k, c in items:
        # Si k est inclus dans un élément déjà 'kept' d'au moins même ordre de fréquence (avec marge min_gain), on jette
        drop = False
        for big, cbig in kept:
            if cbig >= c - min_gain and _is_token_subspan(big, k):
                drop = True
                break
        if not drop:
            kept.append((k, c))
    # Reconstruire un Counter final
    pruned = Counter()
    for k, c in kept:
        pruned[k] = c
    return pruned





# --- Core Logic ---
def build_candidates(
    text: str,
    antidico: set[str],
    antidico_internal: bool = False,
    min_len: int = 3,
    max_len: int = 8,
    require_particle: bool = False,
    ) -> Counter:
    """Builds a cleaner list L by filtering candidates at the source."""
    tokens = tokenize(text)
    cap = [is_capitalized(t) for t in tokens]
    starts = sentence_starts(text)
    counts = Counter()
# 1) Unigrams: Filtered immediately with the antidictionary + 'debut de phrase' starters
    for i, (t, is_cap) in enumerate(zip(tokens, cap)):
        if not is_cap:
            continue
        # si 'starter' et c'est le 1er token d'une phrase -> on ignore
        if i in starts and t in CONVERSATIONAL_STARTERS:
            continue
        # AJOUTER : Filtrer les chiffres romains
        if t in ROMAN_NUMERALS:
            continue
        # AJOUTER : Filtrer les articles contractés
        if is_contracted_article(t):
            continue
        
        # AJOUTER : Filtre anti-OCR simple
        if re.match(r"^([A-Z])\1", t):  # JJ..., BB... → bruit OCR fréquent
            continue
        if len(t) <= 2 and t.isupper() and t.lower() not in {"iv", "vi", "ix"}:
            continue
            
        if t.lower() not in antidico:
            counts[t] += 1
    # 2) N-grams: version 'gourmande' PN (particule PN)+, jusqu'à max_len
    n = len(tokens)
    for i in range(n):
        if not cap[i]:
            continue
        # si début de phrase + starter, on saute
        if i in starts and tokens[i] in CONVERSATIONAL_STARTERS:
            continue
        # même hors début de phrase, on ne démarre pas sur un starter
        if tokens[i] in CONVERSATIONAL_STARTERS:
            continue
        # AJOUTER : Filtrer les chiffres romains
        if tokens[i] in ROMAN_NUMERALS:
            continue
        # AJOUTER : Filtrer les articles contractés
        if is_contracted_article(tokens[i]):
            continue
        if is_stop(tokens[i], antidico):
            continue
        phrase = build_ngram_greedy(
            tokens, cap, i,
            min_len=min_len,
            max_len=max_len,
            require_particle=require_particle
        )

        if phrase:
            parts = phrase.split()
            # extrémités non vides
            if is_stop(parts[0], antidico) or is_stop(parts[-1], antidico):
                continue

            # (optionnel) filtrer aussi l'intérieur hors particules
            # -> activer si tu vois encore du bruit
            # core = [w for w in parts if w.lower() not in PARTICLES]
            # if any(is_stop(w, antidico) for w in core):
            #     continue

            counts[phrase] += 1

            # (optionnel) expansions intermédiaires, comme tu le fais déjà
            k = 3
            while k < len(parts):
                sub = " ".join(parts[:k+1])
                counts[sub] += 1
                k += 2

    return counts


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Build a clean candidate list L from text files.")
    parser.add_argument("texts", nargs="+", help="One or more input .txt files.")
    parser.add_argument("--antidico", type=Path, default="resources/antidictionnaire.txt",
                        help="Path to the antidictionary.")
    parser.add_argument("--antidico-internal", action="store_true",
                        help="Filtrer aussi les mots internes (hors particules) contre l'antidico")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory.")
    parser.add_argument("--prune", action="store_true",
                    help="Activer la suppression des sous-chaînes redondantes")
    parser.add_argument("--min-gain", type=int, default=1,
                    help="Marge de fréquence pour garder un sous-n-gram (défaut: 1)")
    parser.add_argument("--spacy-verb-stop", action="store_true",
                    help="Augmenter l'antidico avec les lemmes VERB/AUX/INTJ (et impératifs) détectés par spaCy")
    parser.add_argument("--spacy-model", default="fr_core_news_md",
                    help="Modèle spaCy à utiliser (par défaut: fr_core_news_md)")
    parser.add_argument("--require-particle", action="store_true",
                    help="Le n-gram doit contenir au moins une particule (de, du, des...)")
    parser.add_argument("--min-len", type=int, default=3,
                    help="Longueur minimale des n-grammes à conserver (défaut: 3)")
    parser.add_argument("--max-len", type=int, default=8,
                    help="Longueur maximale des n-grammes à générer (défaut: 8)")
    parser.add_argument("--postfilter-pos", action="store_true",
                    help="Filtrer les candidats finaux par POS (retire ADJ/ADV/VERB-Part unigrams et VERB-PRON)")

    args = parser.parse_args()

    # Load resources
    antidictionary = load_antidictionary(args.antidico)

    # Load and process corpus
  # Load and process corpus
    print("Reading and normalizing corpus files...")
    text_parts = [Path(p).read_text(encoding="utf-8") for p in args.texts]
    raw_text = "\n".join(text_parts)
    full_text = normalize(raw_text)


# Build candidate list
    print("Building candidate list L...")
    augmented_antidico = antidictionary
    if args.spacy_verb_stop:
        print("Building dynamic verb/pragmatic stoplist with spaCy...")
        dyn = spacy_dynamic_antidico(full_text, model=args.spacy_model)
        augmented_antidico = set(antidictionary) | dyn

    print("Building candidate list L...")

    counts = build_candidates(
        full_text,
        augmented_antidico,
        antidico_internal=args.antidico_internal,
        min_len=args.min_len,
        max_len=args.max_len,
        require_particle=args.require_particle,
    )



    if args.prune:
        print("Pruning overlapping substrings...")
        before = len(counts)
        counts = prune_substrings(counts, min_gain=args.min_gain)
        after = len(counts)
        print(f"Pruned {before - after} entries (kept {after}).")

    # Post-filtre POS si demandé
    if args.postfilter_pos:
        print("Applying POS post-filter...")
        try:
            import spacy
            nlp_pf = spacy.load(args.spacy_model, disable=["ner"])
        except Exception as e:
            print(f"[postfilter_pos skipped] {e}")
            nlp_pf = None
        before_pf = len(counts)
        counts = postfilter_pos(counts, nlp_pf)
        after_pf = len(counts)
        print(f"Post-filtered {before_pf - after_pf} entries (kept {after_pf}).")

    # Write output files
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    (outdir / "L.txt").write_text("\n".join([k for k, _ in items]) + "\n", encoding="utf-8")
    with (outdir / "L_counts.tsv").open("w", encoding="utf-8") as f:
        for k, c in items:
            f.write(f"{k}\t{c}\n")

    print(f"\nSuccessfully generated clean list L with {len(counts)} candidates.")
    print(f"Normalized text length: {len(full_text):,} characters")
    print(f"-> {outdir / 'L.txt'}")
    print(f"-> {outdir / 'L_counts.tsv'}")


if __name__ == "__main__":
    main()




    #python3 listL.py corpus_asimov/Fondation_et_empire_sample.txt -o outputs/ --prune --min-gain 1
    #python3 listL.py corpus_asimov/Fondation_et_empire_sample.txt -o outputs/ --prune --min-gain 1 --antidico-internal

#python3 listL.py corpus_asimov/*.txt -o outputs/ \
#--prune --min-gain 2 --antidico-internal \
#--require-particle --min-len 3 \
#--spacy-verb-stop --postfilter-pos


