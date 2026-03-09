#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_submission.py — V5 HYBRIDE (Auto + Sécurité Manuelle)
"""

import pandas as pd
import networkx as nx
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

BOOK_CODES = {
    "prelude_a_fondation": "paf",
    "les_cavernes_d_acier": "lca"
}

# On réduit la fenêtre pour augmenter la PRÉCISION
WINDOW_SIZE = 120           
MIN_WEIGHT_THRESHOLD = 3

# --- 1. LA LISTE DE SÉCURITÉ (MANUELLE) ---
# Ces règles écrasent l'automatisme pour garantir le score sur les héros.
# C'est ce qui vous manquait dans la version 0.32.
CRITICAL_ALIASES = {
    # Héros Cavernes d'Acier
    "Baley": "Lije Baley", "Lije": "Lije Baley", "Elijah": "Lije Baley",
    "Jessie": "Jessie Baley",
    "Bentley": "Bentley Baley",
    "Daneel": "R. Daneel Olivaw", "Olivaw": "R. Daneel Olivaw", "R. Daneel": "R. Daneel Olivaw",
    "Giskard": "R. Giskard Reventlov", "Reventlov": "R. Giskard Reventlov",
    "Fastolfe": "Han Fastolfe", "Han": "Han Fastolfe",
    "Enderby": "Julius Enderby", "Julius": "Julius Enderby",
    "Sarton": "Roj Nemennuh Sarton", "Roj": "Roj Nemennuh Sarton",
    "Clousarr": "Francis Clousarr", "Francis": "Francis Clousarr",
    
    # Héros Prélude
    "Seldon": "Hari Seldon", "Hari": "Hari Seldon",
    "Dors": "Dors Venabili", "Venabili": "Dors Venabili",
    "Cléon": "Cléon Ier", "Empereur": "Cléon Ier", "Sire": "Cléon Ier",
    "Demerzel": "Eto Demerzel", "Eto": "Eto Demerzel",
    "Yugo": "Yugo Amaryl", "Amaryl": "Yugo Amaryl",
    "Raych": "Raych Seldon",
    "Hummin": "Chetter Hummin", "Chetter": "Chetter Hummin",
    "Randa": "Kiangtow Randa",
    "Rashelle": "Rashelle of Wye", "Wye": "Rashelle of Wye"
}

# Mots à ignorer (Bruit)
GRAPH_BLACKLIST = {
    "Jésus", "Shakespeare", "Heisenberg", "Churchill", "Ahab", 
    "Job", "Naboth", "Jéhu", "Jéhoram", "Noé", "Moïse", 
    "Anciens", "Médiévaliste", "Médiévalistes", "Galactica", "Encyclopaedia",
    "Ciel", "Dieu", "Seigneur", "Quarantecinq", "Jenarr", "Leggen"
}

TITLES_TO_STRIP = {"Dr", "Docteur", "Maire", "Commissaire", "Mme", "M.", "Maître", "Sire", "Empereur", "R.", "Robot", "Général"}

# Chemin vers les ressources de sentiment
_RESOURCES_DIR = Path(__file__).parent / "resources"
FEEL_LEXICON_PATH    = _RESOURCES_DIR / "feel_lexicon.csv"
RELATION_VERBS_PATH  = _RESOURCES_DIR / "relation_verbs_asimov.csv"

# Marqueurs de négation française
NEGATION_MARKERS = {
    "ne", "n", "pas", "jamais", "aucun", "aucune", "sans",
    "ni", "non", "plus", "guère", "nullement", "point"
}

# Intensificateurs (multiplient le score)
INTENSIFIERS = {
    "très", "vraiment", "absolument", "profondément", "sincèrement",
    "terriblement", "extrêmement", "totalement", "complètement", "fortement"
}

# =============================================================================
# CHARGEMENT DES LEXIQUES
# =============================================================================

def load_feel_lexicon(path: Path) -> Dict[str, Tuple[str, int]]:
    """
    Charge le lexique FEEL.
    Retourne un dict: mot -> (polarite, intensite)
    """
    lexicon = {}
    if not path.exists():
        print(f"[AVERTISSEMENT] Lexique FEEL introuvable : {path}")
        return lexicon
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                mot, polarite, intensite = parts[0].strip(), parts[1].strip(), parts[2].strip()
                try:
                    lexicon[mot] = (polarite, int(intensite))
                except ValueError:
                    pass
    print(f"  Lexique FEEL chargé : {len(lexicon)} entrées")
    return lexicon


def load_relation_verbs(path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Charge les verbes de relation directionnels.
    Retourne (verbes_positifs, verbes_négatifs).
    """
    pos_verbs, neg_verbs = set(), set()
    if not path.exists():
        print(f"[AVERTISSEMENT] Verbes de relation introuvables : {path}")
        return pos_verbs, neg_verbs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 2:
                mot, sens = parts[0].strip(), parts[1].strip()
                if sens == "pos":
                    pos_verbs.add(mot)
                elif sens == "neg":
                    neg_verbs.add(mot)
    print(f"  Verbes rel. chargés : {len(pos_verbs)} positifs, {len(neg_verbs)} négatifs")
    return pos_verbs, neg_verbs


# =============================================================================
# ANALYSE DE SENTIMENT AVEC NÉGATION
# =============================================================================

def score_window(
    window_text: str,
    feel: Dict[str, Tuple[str, int]],
    pos_verbs: Set[str],
    neg_verbs: Set[str]
) -> Tuple[int, int]:
    """
    Analyse le texte d'une fenêtre de co-occurrence.
    Gère :
      - Le lexique FEEL (polarité + intensité)
      - Les verbes de relation directionnels
      - La négation (inversion de polarité sur les 3 mots suivants)
      - Les intensificateurs (bonus +1)

    Retourne (score_positif, score_négatif).
    """
    # On tokenise EN GARDANT les ponctuations de fin de phrase comme sentinelles
    raw_tokens = re.findall(r"[a-zàâçéèêëîïôùûüæœ']+|[.!?]", window_text.lower())
    pos_total, neg_total = 0, 0
    neg_countdown  = 0   # nombre de mots encore sous l'effet d'une négation
    boost_next     = 0   # bonus intensificateur pour le prochain mot lexical

    for token in raw_tokens:
        # Fin de phrase → la négation ne peut pas déborder sur la phrase suivante
        if token in ('.', '!', '?'):
            neg_countdown = 0
            boost_next    = 0
            continue

        # Détection d'un intensificateur
        if token in INTENSIFIERS:
            boost_next = 1
            continue

        # Détection d'un marqueur de négation
        if token in NEGATION_MARKERS:
            neg_countdown = 4   # les 4 prochains tokens lexicaux sont inversés
            boost_next = 0
            continue

        flip = neg_countdown > 0
        if neg_countdown > 0:
            neg_countdown -= 1

        # Score via le lexique FEEL
        if token in feel:
            polarite, intensite = feel[token]
            score = intensite + boost_next
            boost_next = 0
            if polarite == "negative":
                score = -score
            if flip:
                score = -score
            if score > 0:
                pos_total += score
            else:
                neg_total += abs(score)
            continue

        boost_next = 0

        # Score via les verbes de relation directionnels
        if token in pos_verbs:
            delta = 2
            if flip:
                neg_total += delta
            else:
                pos_total += delta
        elif token in neg_verbs:
            delta = 2
            if flip:
                pos_total += delta
            else:
                neg_total += delta

    return pos_total, neg_total


def classify_relation(pos: int, neg: int) -> str:
    """
    Classifie la relation entre deux personnages :
      'pour'   si les signaux positifs dominent (majorité simple)
      'contre' si les signaux négatifs dominent
      'neutre' si équilibre strict ou absence de signal
    """
    if pos == 0 and neg == 0:
        return "neutre"
    if pos > neg:
        return "pour"
    if neg > pos:
        return "contre"
    return "neutre"


# =============================================================================
# OUTILS
# =============================================================================

def normalize_name(name):
    """Normalise la casse."""
    parts = name.title().split()
    fixed_parts = []
    for p in parts:
        if re.match(r'^(Ier|Ii|Iii|Iv|V|Vi|Vii|Ix|X)$', p, re.IGNORECASE):
            if p.lower() == 'ier': fixed_parts.append('Ier')
            else: fixed_parts.append(p.upper())
        elif p.lower() in ["de", "la", "von", "van", "of"]:
            fixed_parts.append(p.lower())
        else:
            fixed_parts.append(p)
    return " ".join(fixed_parts)

def clean_for_matching(name):
    parts = name.split()
    if not parts: return ""
    if parts[0] in TITLES_TO_STRIP and len(parts) > 1:
        return " ".join(parts[1:])
    return name

def build_hybrid_alias_map(lp_list, corpus_dir):
    print("Construction Hybride des alias...")
    
    # 1. Scan du corpus pour les fréquences
    full_text = ""
    for folder_name in BOOK_CODES:
        folder_path = corpus_dir / folder_name
        if folder_path.exists():
            for f in folder_path.glob("*.txt.preprocessed"):
                full_text += f.read_text(encoding="utf-8") + "\n"
    
    normalized_lp = {normalize_name(n) for n in lp_list if n not in GRAPH_BLACKLIST}
    
    freqs = Counter()
    for name in normalized_lp:
        count = len(re.findall(r'\b' + re.escape(name) + r'\b', full_text, re.IGNORECASE))
        freqs[name] = count

    sorted_candidates = sorted(normalized_lp, key=lambda x: (freqs[x], len(x)), reverse=True)
    alias_map = {}
    
    # 2. AUTOMATISATION (Pour les persos secondaires)
    for name in sorted_candidates:
        alias_map[name] = name
        
    for child in sorted_candidates:
        child_clean = clean_for_matching(child)
        best_parent = child
        best_parent_score = -1
        
        for potential_parent in sorted_candidates:
            if child == potential_parent: continue
            parent_clean = clean_for_matching(potential_parent)
            
            # Inclusion stricte
            if re.search(r'\b' + re.escape(child_clean) + r'\b', parent_clean):
                if freqs[potential_parent] > best_parent_score:
                    best_parent = potential_parent
                    best_parent_score = freqs[potential_parent]
        
        if best_parent != child:
            alias_map[child] = best_parent

    # 3. FORCE BRUTE MANUELLE (Pour écraser les erreurs de l'auto)
    # C'est ici qu'on sauve le score
    print("Application des correctifs manuels...")
    for short, target in CRITICAL_ALIASES.items():
        norm_short = normalize_name(short)
        norm_target = normalize_name(target)
        
        # On force le mapping
        alias_map[norm_short] = norm_target
        # On s'assure que la cible est bien une clé canonique (pointe vers elle-même)
        alias_map[norm_target] = norm_target

    # 4. Identification des Vitaux (Top 20 après fusion)
    final_counts = Counter()
    for name, count in freqs.items():
        # Attention : on utilise la map mise à jour
        if name in alias_map:
            canon = alias_map[name]
            final_counts[canon] += count
    
    # On sauve les 20 plus fréquents même s'ils sont isolés
    vital_chars = {name for name, _ in final_counts.most_common(20)}
    
    return alias_map, vital_chars

# =============================================================================
# LISSAGE GLOBAL DES RELATIONS
# =============================================================================

def smooth_relations_globally(df_dict, min_chapters=3, min_confidence=0.60):
    """
    Post-traitement : pour les paires (A,B) apparaissant dans au moins
    min_chapters chapitres avec une relation dominante à >= min_confidence,
    on force cette relation dans TOUS les chapitres.

    Principe : si Dors/Hari est « pour » dans 9 chapitres sur 16,
    les 7 « contre » sont probablement des artefacts de fenêtres
    chargées en mots d'action (scènes de combat, danger).
    """
    import xml.etree.ElementTree as ET
    from collections import Counter as Ctr

    _NS = 'http://graphml.graphdrawing.org/xmlns'

    # 1. Collecter toutes les relations par paire à travers tous les chapitres
    pair_rels = {}   # (A, B) -> [rel, rel, ...]
    for graphml_str in df_dict["graphml"]:
        try:
            root = ET.fromstring(graphml_str)
        except Exception:
            continue
        keys = {k.get('id'): k.get('attr.name')
                for k in root.iter(f'{{{_NS}}}key')}
        for edge in root.iter(f'{{{_NS}}}edge'):
            src  = edge.get('source')
            tgt  = edge.get('target')
            pair = tuple(sorted([src, tgt]))
            data = {keys.get(d.get('key')): d.text
                    for d in edge.iter(f'{{{_NS}}}data')}
            rel  = data.get('relation', 'neutre')
            pair_rels.setdefault(pair, []).append(rel)

    # 2. Déterminer la relation stable pour les paires qualifiées
    overrides = {}
    for pair, rels in pair_rels.items():
        if len(rels) < min_chapters:
            continue
        counts     = Ctr(rels)
        top, count = counts.most_common(1)[0]
        if count / len(rels) >= min_confidence:
            overrides[pair] = top

    print(f"  Lissage global : {len(overrides)} paires stabilisées "
          f"(≥{min_chapters} chap., ≥{int(min_confidence*100)}% dominance)")

    # 3. Appliquer les overrides dans chaque GraphML
    new_graphmls = []
    for graphml_str in df_dict["graphml"]:
        try:
            root = ET.fromstring(graphml_str)
        except Exception:
            new_graphmls.append(graphml_str)
            continue
        keys     = {k.get('id'): k.get('attr.name')
                    for k in root.iter(f'{{{_NS}}}key')}
        key_inv  = {v: k for k, v in keys.items()}
        rel_key  = key_inv.get('relation')

        if rel_key:
            for edge in root.iter(f'{{{_NS}}}edge'):
                src  = edge.get('source')
                tgt  = edge.get('target')
                pair = tuple(sorted([src, tgt]))
                if pair in overrides:
                    for d in edge.iter(f'{{{_NS}}}data'):
                        if d.get('key') == rel_key:
                            d.text = overrides[pair]

        new_graphmls.append(ET.tostring(root, encoding='unicode'))

    df_dict["graphml"] = new_graphmls
    return df_dict


# =============================================================================
# MOTEUR DE GRAPHE
# =============================================================================

def get_entities_positions(text, alias_map):
    search_terms = sorted(alias_map.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, search_terms)) + r')\b', re.IGNORECASE)
    
    matches = []
    for match in pattern.finditer(text):
        start, _ = match.span()
        raw_found = match.group()
        norm_found = normalize_name(raw_found)
        
        if norm_found in alias_map:
            canon = alias_map[norm_found]
            word_idx = text[:start].count(' ')
            matches.append((word_idx, raw_found, canon))
            
    matches.sort(key=lambda x: (x[0], -len(x[1])))
    
    final_entities = []
    last_idx = -1
    for idx, raw, canon in matches:
        if idx > last_idx:
            final_entities.append((idx, raw, canon))
            last_idx = idx 
    return final_entities

def build_graph_for_chapter(text, alias_map, vital_chars, feel, pos_verbs, neg_verbs):
    G = nx.Graph()
    entities = get_entities_positions(text, alias_map)
    node_variants = {}
    # Accumulation des scores de sentiment par paire
    edge_scores = {}   # (canonA, canonB) -> [pos_total, neg_total]

    words = text.split()   # pour extraire les fenêtres de texte brut

    for i in range(len(entities)):
        curr_idx, curr_raw, curr_canon = entities[i]

        if curr_canon not in node_variants: node_variants[curr_canon] = set()
        node_variants[curr_canon].add(curr_raw)

        for j in range(i + 1, len(entities)):
            next_idx, next_raw, next_canon = entities[j]
            if next_canon not in node_variants: node_variants[next_canon] = set()
            node_variants[next_canon].add(next_raw)

            if (next_idx - curr_idx) > WINDOW_SIZE: break
            if curr_canon == next_canon: continue

            # ── Analyse sentimentale de la fenêtre ──────────────────────────
            start_w = max(0, curr_idx)
            end_w   = min(len(words), next_idx + len(next_raw.split()))
            window_text = " ".join(words[start_w:end_w])
            pos, neg = score_window(window_text, feel, pos_verbs, neg_verbs)

            edge_key = tuple(sorted([curr_canon, next_canon]))
            if edge_key not in edge_scores:
                edge_scores[edge_key] = [0, 0]
            edge_scores[edge_key][0] += pos
            edge_scores[edge_key][1] += neg
            # ────────────────────────────────────────────────────────────────

            if G.has_edge(curr_canon, next_canon):
                G[curr_canon][next_canon]['weight'] += 1
            else:
                G.add_edge(curr_canon, next_canon, weight=1)

    # Appliquer le type de relation sur chaque arête
    for (u, v), (pos_total, neg_total) in edge_scores.items():
        if G.has_edge(u, v):
            G[u][v]['relation'] = classify_relation(pos_total, neg_total)

    # Valeur par défaut pour les arêtes sans données de sentiment
    for u, v in G.edges():
        if 'relation' not in G[u][v]:
            G[u][v]['relation'] = 'neutre'

    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        if data['weight'] < MIN_WEIGHT_THRESHOLD:
            edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)

    for canon, variants in node_variants.items():
        if not G.has_node(canon): G.add_node(canon)
        variants.add(canon) 
        G.nodes[canon]["names"] = ";".join(sorted(list(variants)))

    # On ne garde les isolés que s'ils sont dans le TOP 20 vital
    isolates = list(nx.isolates(G))
    for node in isolates:
        if node not in vital_chars:
            G.remove_node(node)
            
    return G

# =============================================================================
# MAIN
# =============================================================================

def debug_pair(text, alias_map, feel, pos_verbs, neg_verbs, charA, charB):
    """
    Affiche les mots qui contribuent au score de la relation entre charA et charB.
    """
    entities = get_entities_positions(text, alias_map)
    words = text.split()
    windows_found = 0

    for i in range(len(entities)):
        ci, _, cA = entities[i]
        if cA != charA: continue
        for j in range(i + 1, len(entities)):
            nj, _, cB = entities[j]
            if (nj - ci) > WINDOW_SIZE: break
            if cB != charB: continue

            windows_found += 1
            start_w = max(0, ci)
            end_w   = min(len(words), nj + 5)
            window_text = " ".join(words[start_w:end_w])
            tokens = re.findall(r"[a-zàâçéèêëîïôùûüæœ']+|[.!?]", window_text.lower())

            hits = []
            neg_cd = 0
            for t in tokens:
                if t in ('.', '!', '?'):
                    neg_cd = 0; continue
                flip = neg_cd > 0
                if t in NEGATION_MARKERS:
                    neg_cd = 4; hits.append(f"[NEG:{t}]"); continue
                if neg_cd > 0: neg_cd -= 1
                if t in feel:
                    pol, intens = feel[t]
                    sign = "+" if (pol == "positive") != flip else "-"
                    hits.append(f"{sign}FEEL({t},{intens})")
                elif t in pos_verbs:
                    sign = "+" if not flip else "-"
                    hits.append(f"{sign}VRB_POS({t})")
                elif t in neg_verbs:
                    sign = "-" if not flip else "+"
                    hits.append(f"{sign}VRB_NEG({t})")

            pos, neg = score_window(window_text, feel, pos_verbs, neg_verbs)
            print(f"  Fenêtre {windows_found} [{start_w}-{end_w}] : pos={pos} neg={neg} → {classify_relation(pos,neg)}")
            print(f"    Mots détectés : {', '.join(hits) if hits else '(aucun)'}")
            print(f"    Texte : {window_text[:200]}")
            print()

    if windows_found == 0:
        print(f"  Aucune fenêtre commune trouvée entre '{charA}' et '{charB}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LP", default="outputs/LP_final.txt", type=Path)
    parser.add_argument("--corpus", default="corpus_asimov_leaderboard", type=Path)
    parser.add_argument("-o", "--output", default="my_submission.csv", type=Path)
    parser.add_argument("--debug-pair", nargs=3, metavar=("CHAPITRE_ID", "PERSO_A", "PERSO_B"),
                        help="Ex: paf0 'Hari Seldon' 'Chetter Hummin'")
    args = parser.parse_args()

    print("--- Génération HYBRIDE (V5) + Analyse de Relation ---")

    if not args.LP.exists(): return
    lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]

    # Chargement des lexiques de sentiment
    feel      = load_feel_lexicon(FEEL_LEXICON_PATH)
    pos_verbs, neg_verbs = load_relation_verbs(RELATION_VERBS_PATH)

    # ── Mode DEBUG ────────────────────────────────────────────────────────────
    if args.debug_pair:
        chap_id, perso_a, perso_b = args.debug_pair
        if not args.LP.exists(): return
        lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]
        alias_map, vital_chars = build_hybrid_alias_map(lp_list, args.corpus)

        book_code = chap_id[:3]          # ex: 'paf'
        chap_num  = int(chap_id[3:]) + 1  # ex: 0 -> chapter_1
        folder_name = {v: k for k, v in BOOK_CODES.items()}.get(book_code)
        if not folder_name:
            print(f"Code livre inconnu: {book_code}"); return
        fpath = args.corpus / folder_name / f"chapter_{chap_num}.txt.preprocessed"
        if not fpath.exists():
            print(f"Fichier introuvable: {fpath}"); return

        text = fpath.read_text(encoding="utf-8")
        print(f"\n=== DEBUG relation : '{perso_a}' ↔ '{perso_b}' (chapitre {chap_id}) ===")
        debug_pair(text, alias_map, feel, pos_verbs, neg_verbs, perso_a, perso_b)
        return
    # ─────────────────────────────────────────────────────────────────────────

    # Construction Hybride (Auto + Manuel)
    alias_map, vital_chars = build_hybrid_alias_map(lp_list, args.corpus)

    df_dict = {"ID": [], "graphml": []}
    
    for folder_name, book_code in BOOK_CODES.items():
        folder_path = args.corpus / folder_name
        if not folder_path.exists(): continue
            
        files = sorted(folder_path.glob("chapter_*.txt.preprocessed"), 
                       key=lambda p: int(re.search(r'\d+', p.name).group()))
        
        print(f"Traitement {book_code}...")
        for fpath in files:
            file_num = int(re.search(r'\d+', fpath.name).group())
            chap_num = max(0, file_num - 1)
            chap_id = f"{book_code}{chap_num}"
            
            text = fpath.read_text(encoding="utf-8")
            G = build_graph_for_chapter(text, alias_map, vital_chars, feel, pos_verbs, neg_verbs)
            
            graphml_str = "".join(nx.generate_graphml(G))
            df_dict["ID"].append(chap_id)
            df_dict["graphml"].append(graphml_str)

    # ─ Post-traitement : lissage global des relations ───────────────────
    print("Post-traitement des relations...")
    df_dict = smooth_relations_globally(df_dict, min_chapters=4, min_confidence=0.50)
    # ─────────────────────────────────────────────────────

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv(args.output)
    print(f"Terminé : {args.output}")

if __name__ == "__main__":
    main()