#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_submission.py — V8 ROLLBACK (Focus sur le Rappel pour maximiser le score Kaggle)
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

# 1. Fenêtre glissante ajustée au "Sweet Spot" (150 mots)
MAX_WINDOW = 150 
# 2. On garde TOUTES les relations trouvées (Seuil à 1)
MIN_WEIGHT_THRESHOLD = 1 

CRITICAL_ALIASES = {
    # Les Baley
    "Baley": "Lije Baley", "Lije": "Lije Baley", "Elijah": "Lije Baley",
    "Jessie": "Jessie Baley", "Bentley": "Bentley Baley",
    
    # Les Spaciens / Robots
    "Daneel": "R. Daneel Olivaw", "Olivaw": "R. Daneel Olivaw", "R. Daneel": "R. Daneel Olivaw",
    "Giskard": "R. Giskard Reventlov", "Reventlov": "R. Giskard Reventlov",
    "Fastolfe": "Han Fastolfe", "Han": "Han Fastolfe",
    "Sarton": "Roj Nemennuh Sarton", "Roj": "Roj Nemennuh Sarton",
    "Enderby": "Julius Enderby", "Julius": "Julius Enderby",
    
    # Les Héros Fondation
    "Seldon": "Hari Seldon", "Hari": "Hari Seldon",
    "Dors": "Dors Venabili", "Venabili": "Dors Venabili",
    "Cléon": "Cléon Ier", "Empereur": "Cléon Ier", "Sire": "Cléon Ier",
    "Demerzel": "Eto Demerzel", "Eto": "Eto Demerzel",
    "Amaryl": "Yugo Amaryl", "Yugo": "Yugo Amaryl",
    "Raych": "Raych Seldon"
}

GRAPH_BLACKLIST = {
    "Jésus", "Shakespeare", "Heisenberg", "Churchill", "Ahab", 
    "Job", "Naboth", "Jéhu", "Jéhoram", "Noé", "Moïse", 
    "Anciens", "Médiévaliste", "Médiévalistes", "Galactica", "Encyclopaedia",
    "Ciel", "Dieu", "Seigneur", "Quarantecinq",
}

TITLES_TO_STRIP = {"Dr", "Docteur", "Maire", "Commissaire", "Mme", "M.", "Maître", "Sire", "Empereur", "R.", "Robot", "Général"}

# Chemins vers les ressources de sentiment
_RESOURCES_DIR = Path(__file__).parent / "resources"
FEEL_LEXICON_PATH   = _RESOURCES_DIR / "feel_lexicon.csv"
RELATION_VERBS_PATH = _RESOURCES_DIR / "relation_verbs_asimov.csv"

# Marqueurs de négation française
NEGATION_MARKERS = {
    "ne", "n", "pas", "jamais", "aucun", "aucune", "sans",
    "ni", "non", "plus", "guère", "nullement", "point"
}

# Intensificateurs (bonus +1 sur le mot suivant)
INTENSIFIERS = {
    "très", "vraiment", "absolument", "profondément", "sincèrement",
    "terriblement", "extrêmement", "totalement", "complètement", "fortement"
}

# =============================================================================
# OUTILS
# =============================================================================

def normalize_name(name):
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

# =============================================================================
# CHARGEMENT DES LEXIQUES DE SENTIMENT
# =============================================================================

def load_feel_lexicon(path: Path) -> Dict[str, Tuple[str, int]]:
    """Charge le lexique FEEL. Retourne un dict: mot -> (polarite, intensite)"""
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
    """Charge les verbes de relation directionnels. Retourne (pos_verbs, neg_verbs)."""
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
    Analyse le sentiment d'une fenêtre de co-occurrence.
    Gère : lexique FEEL, verbes directionnels, négation, intensificateurs.
    Retourne (score_positif, score_négatif).
    """
    raw_tokens = re.findall(r"[a-zàâçéèêëîïôùûüæœ']+|[.!?]", window_text.lower())
    pos_total, neg_total = 0, 0
    neg_countdown = 0
    boost_next    = 0

    for token in raw_tokens:
        # Fin de phrase → réinitialiser la négation (pas de débordement inter-phrase)
        if token in ('.', '!', '?'):
            neg_countdown = 0
            boost_next    = 0
            continue

        if token in INTENSIFIERS:
            boost_next = 1
            continue

        if token in NEGATION_MARKERS:
            neg_countdown = 4
            boost_next = 0
            continue

        flip = neg_countdown > 0
        if neg_countdown > 0:
            neg_countdown -= 1

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
    'pour' si pos > neg, 'contre' si neg > pos, 'neutre' sinon.
    """
    if pos == 0 and neg == 0:
        return "neutre"
    if pos > neg:
        return "pour"
    if neg > pos:
        return "contre"
    return "neutre"


def build_hybrid_alias_map(lp_list, corpus_dir):
    print("Construction Hybride des alias...")
    
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
    
    for name in sorted_candidates:
        alias_map[name] = name
        
    for child in sorted_candidates:
        child_clean = clean_for_matching(child)
        best_parent = child
        best_parent_score = -1
        
        for potential_parent in sorted_candidates:
            if child == potential_parent: continue
            parent_clean = clean_for_matching(potential_parent)
            
            if re.search(r'\b' + re.escape(child_clean) + r'\b', parent_clean):
                if freqs[potential_parent] > best_parent_score:
                    best_parent = potential_parent
                    best_parent_score = freqs[potential_parent]
        
        if best_parent != child:
            alias_map[child] = best_parent

    print("Application des correctifs manuels...")
    for short, target in CRITICAL_ALIASES.items():
        norm_short = normalize_name(short)
        norm_target = normalize_name(target)
        
        alias_map[norm_short] = norm_target
        alias_map[norm_target] = norm_target

    final_counts = Counter()
    for name, count in freqs.items():
        if name in alias_map:
            canon = alias_map[name]
            final_counts[canon] += count
    
    # 3. Filet de sauvetage maximum : on garde le Top 50 pour ne rater personne
    vital_chars = {name for name, _ in final_counts.most_common(50)}
    
    return alias_map, vital_chars

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
    edge_scores   = {}   # (canonA, canonB) -> [pos_total, neg_total]

    words = text.split()  # pour extraire les fenêtres de texte

    for i in range(len(entities)):
        curr_idx, curr_raw, curr_canon = entities[i]

        if curr_canon not in node_variants: node_variants[curr_canon] = set()
        node_variants[curr_canon].add(curr_raw)

        for j in range(i + 1, len(entities)):
            next_idx, next_raw, next_canon = entities[j]
            distance = next_idx - curr_idx

            if distance > MAX_WINDOW: break
            if curr_canon == next_canon: continue

            if next_canon not in node_variants: node_variants[next_canon] = set()
            node_variants[next_canon].add(next_raw)

            # Poids de co-occurrence : 1 rencontre = +1 (logique V8 inchangée)
            if G.has_edge(curr_canon, next_canon):
                G[curr_canon][next_canon]['weight'] += 1
            else:
                G.add_edge(curr_canon, next_canon, weight=1)

            # Analyse sentimentale de la fenêtre
            start_w = max(0, curr_idx)
            end_w   = min(len(words), next_idx + len(next_raw.split()))
            window_text = " ".join(words[start_w:end_w])
            pos, neg = score_window(window_text, feel, pos_verbs, neg_verbs)

            edge_key = tuple(sorted([curr_canon, next_canon]))
            if edge_key not in edge_scores:
                edge_scores[edge_key] = [0, 0]
            edge_scores[edge_key][0] += pos
            edge_scores[edge_key][1] += neg

    # Appliquer le type de relation sur chaque arête
    for (u, v), (pos_total, neg_total) in edge_scores.items():
        if G.has_edge(u, v):
            G[u][v]['relation'] = classify_relation(pos_total, neg_total)

    # Valeur par défaut pour les arêtes sans données de sentiment
    for u, v in G.edges():
        if 'relation' not in G[u][v]:
            G[u][v]['relation'] = 'neutre'

    # Filtrage — seuil à 1 (garde tout ce qui s'est croisé au moins une fois)
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data['weight'] < MIN_WEIGHT_THRESHOLD]
    G.remove_edges_from(edges_to_remove)

    for canon, variants in node_variants.items():
        if not G.has_node(canon): G.add_node(canon)
        variants.add(canon)
        G.nodes[canon]["names"] = ";".join(sorted(list(variants)))

    # Nettoyage des nœuds isolés, sauf s'ils font partie du Top 50
    isolates = list(nx.isolates(G))
    for node in isolates:
        if node not in vital_chars:
            G.remove_node(node)

    return G

# =============================================================================
# LISSAGE GLOBAL DES RELATIONS
# =============================================================================

def smooth_relations_globally(df_dict, min_chapters=4, min_confidence=0.50):
    """
    Post-traitement : pour les paires (A,B) apparaissant dans au moins
    min_chapters chapitres avec une relation dominante à >= min_confidence,
    on force cette relation dans TOUS les chapitres.
    """
    import xml.etree.ElementTree as ET
    from collections import Counter as Ctr

    _NS = 'http://graphml.graphdrawing.org/xmlns'
    # Enregistrer les namespaces pour éviter les préfixes ns0: dans la sortie
    ET.register_namespace('', _NS)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

    pair_rels = {}
    for graphml_str in df_dict["graphml"]:
        try:
            root = ET.fromstring(graphml_str)
        except Exception:
            continue
        keys = {k.get('id'): k.get('attr.name') for k in root.iter(f'{{{_NS}}}key')}
        for edge in root.iter(f'{{{_NS}}}edge'):
            src  = edge.get('source')
            tgt  = edge.get('target')
            pair = tuple(sorted([src, tgt]))
            data = {keys.get(d.get('key')): d.text for d in edge.iter(f'{{{_NS}}}data')}
            rel  = data.get('relation', 'neutre')
            pair_rels.setdefault(pair, []).append(rel)

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

    new_graphmls = []
    for graphml_str in df_dict["graphml"]:
        try:
            root = ET.fromstring(graphml_str)
        except Exception:
            new_graphmls.append(graphml_str)
            continue
        keys    = {k.get('id'): k.get('attr.name') for k in root.iter(f'{{{_NS}}}key')}
        key_inv = {v: k for k, v in keys.items()}
        rel_key = key_inv.get('relation')

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
# DÉBOGAGE
# =============================================================================

def debug_pair(text, alias_map, feel, pos_verbs, neg_verbs, charA, charB):
    """Affiche les mots qui contribuent au score de la relation entre charA et charB."""
    entities = get_entities_positions(text, alias_map)
    words = text.split()
    windows_found = 0

    for i in range(len(entities)):
        ci, _, cA = entities[i]
        if cA != charA: continue
        for j in range(i + 1, len(entities)):
            nj, _, cB = entities[j]
            if (nj - ci) > MAX_WINDOW: break
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
            print(f"  Fenêtre {windows_found} [{start_w}-{end_w}] : pos={pos} neg={neg} → {classify_relation(pos, neg)}")
            print(f"    Mots détectés : {', '.join(hits) if hits else '(aucun)'}")
            print(f"    Texte : {window_text[:200]}")
            print()

    if windows_found == 0:
        print(f"  Aucune fenêtre commune trouvée entre '{charA}' et '{charB}'")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LP", default="outputs/LP_final.txt", type=Path)
    parser.add_argument("--corpus", default="corpus_asimov_leaderboard", type=Path)
    parser.add_argument("-o", "--output", default="my_submission.csv", type=Path)
    parser.add_argument("--debug-pair", nargs=3, metavar=("CHAPITRE_ID", "PERSO_A", "PERSO_B"),
                        help="Ex: paf0 'Hari Seldon' 'Chetter Hummin'")
    args = parser.parse_args()

    print("--- Génération HYBRIDE V8 + Analyse de Relation ---")

    if not args.LP.exists():
        print("Erreur: Fichier LP introuvable. Avez-vous exécuté listLP.py ?")
        return

    lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]

    # Chargement des lexiques de sentiment
    feel = load_feel_lexicon(FEEL_LEXICON_PATH)
    pos_verbs, neg_verbs = load_relation_verbs(RELATION_VERBS_PATH)

    # Mode DEBUG
    if args.debug_pair:
        chap_id, perso_a, perso_b = args.debug_pair
        alias_map, vital_chars = build_hybrid_alias_map(lp_list, args.corpus)
        book_code = chap_id[:3]
        chap_num  = int(chap_id[3:]) + 1
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

    # Post-traitement : lissage global des relations
    print("Post-traitement des relations...")
    df_dict = smooth_relations_globally(df_dict, min_chapters=4, min_confidence=0.50)

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv(args.output)
    print(f"Terminé : {args.output}")

if __name__ == "__main__":
    main()