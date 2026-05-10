#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_submission_DL_only.py — Approche Deep Learning Pure
==========================================================
ÉTUDE COMPARATIVE — Système 1 / 3

Méthode d'évaluation des relations : ZÉRO-SHOT NLI uniquement.
Aucun lexique FEEL ni verbe directionnel n'est utilisé.

Le modèle `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` reçoit chaque
fenêtre de co-occurrence et classe la relation via deux hypothèses :
  - "alliés, coopération, amis"
  - "ennemis, conflit, danger"

Un score continu (alliance - conflit) est calculé pour chaque fenêtre,
puis moyenné sur toutes les fenêtres de la paire avant classification.

Seuils :  score_nli > 0.1  → "pour"
          score_nli < -0.1 → "contre"
          sinon             → "neutre"

Toute l'infrastructure (Alias, Blacklist, Lissage global) est identique
au script de référence `graph_submission_relations.py`.

Usage :
    python3 graph_submission_DL_only.py \
        --LP outputs/LP_final.txt \
        --corpus corpus_asimov_leaderboard \
        -o DL_only.csv
"""

import pandas as pd
import networkx as nx
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import torch
from transformers import pipeline

# =============================================================================
# CONFIGURATION
# =============================================================================

BOOK_CODES = {
    "prelude_a_fondation": "paf",
    "les_cavernes_d_acier": "lca"
}

WINDOW_SIZE          = 120
MIN_WEIGHT_THRESHOLD = 3

# Seuils de classification pour le score NLI (entre -1 et +1)
# Ajuster ici pour affiner le comportement sans toucher au reste du code.
THRESHOLD_POS = -0.15  # Au lieu de +0.05
THRESHOLD_NEG = -0.30  # Au lieu de -0.05

# Labels candidats pour le Zero-Shot.
# Des phrases complètes donnent de meilleures performances sur les modèles MNLI
# (hypothèse naturelle plutôt qu'une simple liste de mots).
NLI_LABELS = [
    "Ces deux personnages s'entraident, sont amis ou alliés.",
    "Ces deux personnages se détestent, sont adversaires ou s'affrontent."
]

# --- LISTE DE SÉCURITÉ (MANUELLE) ---
CRITICAL_ALIASES = {
    # ----- Les Cavernes d'acier (LCA) -----
    "Baley": "Elijah Baley", "Lije": "Elijah Baley", "Lije Baley": "Elijah Baley", "Elijah": "Elijah Baley",
    "Jessie": "Jessica Baley", "Jessie Baley": "Jessica Baley", "Jessica": "Jessica Baley",
    "Bentley": "Bentley Baley",
    "Daneel": "Daneel Olivaw", "Olivaw": "Daneel Olivaw", "R. Daneel": "Daneel Olivaw", "R. Daneel Olivaw": "Daneel Olivaw",
    "Giskard": "Giskard Reventlov", "Reventlov": "Giskard Reventlov", "R. Giskard Reventlov": "Giskard Reventlov",
    "Fastolfe": "Han Fastolfe", "Han": "Han Fastolfe", "Dr Han Fastolfe": "Han Fastolfe", "Dr Fastolfe": "Han Fastolfe",
    "Sarton": "Roj Nemennuh Sarton", "Roj": "Roj Nemennuh Sarton", "Dr Sarton": "Roj Nemennuh Sarton", "Docteur Sarton": "Roj Nemennuh Sarton",
    "Enderby": "Julius Enderby", "Julius": "Julius Enderby",
    "Rachelle": "Rashelle", "Rashelle de Wye": "Rashelle",
    # ----- Prélude à Fondation (PAF) -----
    "Seldon" : "Hari Seldon", "Hari": "Hari Seldon",
    "Dors": "Dors Venabili", "Venabili": "Dors Venabili",
    "Cléon": "Cleon Ier", "Cléon Ier": "Cleon Ier", "Empereur": "Cleon Ier", "Sire": "Cleon Ier",
    "Demerzel": "Eto Demerzel", "Eto": "Eto Demerzel",
    "Amaryl": "Yugo Amaryl", "Yugo": "Yugo Amaryl",
    "Raych": "Raych Seldon",
    "Rittah": "Mere Rittah", "Mère Rittah": "Mere Rittah", "Mother Rittah": "Mere Rittah",
    "Quatorze": "Maitre Quatorze", "Maître Quatorze": "Maitre Quatorze", "Sunmaster": "Maitre Quatorze",
    "Mannix IV": "Mannix IV Kan", "Mannix": "Mannix IV Kan",
    "Hummin": "Chetter Hummin", "Chetter": "Chetter Hummin"
}

GRAPH_BLACKLIST = {
    "Jésus", "Shakespeare", "Heisenberg", "Churchill", "Ahab",
    "Job", "Naboth", "Jéhu", "Jéhoram", "Noé", "Moïse",
    "Anciens", "Médiévaliste", "Médiévalistes", "Galactica", "Encyclopaedia",
    "Ciel", "Dieu", "Seigneur", "Quarantecinq",
    "Nord", "Dahl", "Mycogène", "Mycogéne", "Wye", "Cinq",
    "Trantor", "Terminus", "Empire", "Secteur", "Hélicon", "Astinwald"
}

TITLES_TO_STRIP = {"Dr", "Docteur", "Maire", "Commissaire", "Mme", "M.", "Maître", "Sire", "Empereur", "R.", "Robot", "Général"}

# =============================================================================
# INITIALISATION DU MODÈLE NLI
# =============================================================================

def _init_nli_pipeline():
    """Charge la pipeline Zero-Shot sur le meilleur device disponible."""
    if torch.cuda.is_available():
        device      = 0
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device      = "mps"
        device_name = "Apple MPS"
    else:
        device      = -1
        device_name = "CPU"
    print(f"  [DL_only] NLI Pipeline : chargement sur {device_name}...")
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=device,
    )

NLI_PIPELINE = _init_nli_pipeline()

# =============================================================================
# INFÉRENCE NLI  (remplacée par le batching dans build_graph_for_chapter)
# =============================================================================
# La fonction scalaire get_nli_score n'est plus utilisée dans le flux principal.
# Le scoring se fait en un seul appel batch après la double boucle d'extraction.


def classify_relation(score_nli: float) -> str:
    """
    Classifie la relation à partir du score NLI moyen ∈ [-1, +1].

      'pour'   si score_nli >  NLI_THRESHOLD_POS
      'contre' si score_nli <  NLI_THRESHOLD_NEG
      'neutre' sinon
    """
    if score_nli > NLI_THRESHOLD_POS:
        return "pour"
    if score_nli < NLI_THRESHOLD_NEG:
        return "contre"
    return "neutre"

# =============================================================================
# OUTILS (identiques à la base)
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

def normalize_corpus_path(path: Path) -> Path:
    raw   = str(path)
    match = re.match(r"^//wsl(?:\.localhost|\$)/[^/]+(/.*)$", raw, re.IGNORECASE)
    if match:
        return Path(match.group(1))
    return path

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
        child_clean   = clean_for_matching(child)
        best_parent   = child
        best_parent_score = -1
        for potential_parent in sorted_candidates:
            if child == potential_parent: continue
            parent_clean = clean_for_matching(potential_parent)
            if re.search(r'\b' + re.escape(child_clean) + r'\b', parent_clean):
                if freqs[potential_parent] > best_parent_score:
                    best_parent       = potential_parent
                    best_parent_score = freqs[potential_parent]
        if best_parent != child:
            alias_map[child] = best_parent

    print("Application des correctifs manuels...")
    for short, target in CRITICAL_ALIASES.items():
        norm_short  = normalize_name(short)
        norm_target = normalize_name(target)
        alias_map[norm_short]  = norm_target
        alias_map[norm_target] = norm_target

    final_counts = Counter()
    for name, count in freqs.items():
        if name in alias_map:
            final_counts[alias_map[name]] += count

    vital_chars = {name for name, _ in final_counts.most_common(20)}
    return alias_map, vital_chars

# =============================================================================
# LISSAGE GLOBAL DES RELATIONS (identique à la base)
# =============================================================================

def smooth_relations_globally(df_dict, min_chapters=3, min_confidence=0.60):
    import xml.etree.ElementTree as ET
    from collections import Counter as Ctr

    _NS = 'http://graphml.graphdrawing.org/xmlns'
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
# MOTEUR DE GRAPHE
# =============================================================================

def get_entities_positions(text, alias_map):
    search_terms = sorted(alias_map.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, search_terms)) + r')\b', re.IGNORECASE)
    matches = []
    for match in pattern.finditer(text):
        start, _   = match.span()
        raw_found  = match.group()
        norm_found = normalize_name(raw_found)
        if norm_found in alias_map:
            canon    = alias_map[norm_found]
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


def build_graph_for_chapter(text, alias_map, vital_chars):
    """
    Construit le graphe d'un chapitre en utilisant UNIQUEMENT le modèle NLI.
    Aucun lexique symbolique n'est sollicité.

    Optimisations :
      - Marge de 20 mots autour des entités pour réduire le bruit de fenêtre.
      - Collecte de toutes les fenêtres avant l'inférence, puis appel NLI
        en un seul batch (batch_size=8) pour maximiser le débit GPU.
    """
    # Marge contextuelle ajoutée de chaque côté de la paire d'entités
    MARGIN = 20

    G             = nx.Graph()
    entities      = get_entities_positions(text, alias_map)
    node_variants: Dict[str, set] = {}
    words = text.split()

    # Listes parallèles : texte à scorer ↔ clé d'arête correspondante
    texts_to_score:    List[str]                   = []
    edge_keys_for_texts: List[Tuple[str, str]]     = []

    # ── Phase 1 : parcours des co-occurrences, collecte des fenêtres ────────
    for i in range(len(entities)):
        curr_idx, curr_raw, curr_canon = entities[i]
        if curr_canon not in node_variants:
            node_variants[curr_canon] = set()
        node_variants[curr_canon].add(curr_raw)

        for j in range(i + 1, len(entities)):
            next_idx, next_raw, next_canon = entities[j]
            if next_canon not in node_variants:
                node_variants[next_canon] = set()
            node_variants[next_canon].add(next_raw)

            if (next_idx - curr_idx) > WINDOW_SIZE: break
            if curr_canon == next_canon: continue

            # Fenêtre recentrée avec marge pour moins de bruit
            start_w     = max(0, curr_idx - MARGIN)
            end_w       = min(len(words), next_idx + len(next_raw.split()) + MARGIN)
            window_text = " ".join(words[start_w:end_w])

            # Collecte pour le batch (tronqué à ~512 tokens)
            if window_text.strip():
                texts_to_score.append(window_text[:1500])
                edge_key = tuple(sorted([curr_canon, next_canon]))
                edge_keys_for_texts.append(edge_key)

            # Comptage du poids de l'arête (indépendant du scoring NLI)
            if G.has_edge(curr_canon, next_canon):
                G[curr_canon][next_canon]['weight'] += 1
            else:
                G.add_edge(curr_canon, next_canon, weight=1)
    # ────────────────────────────────────────────────────────────────────────

    # ── Phase 2 : inférence NLI en un seul appel batch ──────────────────────
    edge_scores: Dict[Tuple[str, str], List[float]] = {}
    if texts_to_score:
        results = NLI_PIPELINE(
            texts_to_score,
            NLI_LABELS,
            multi_label=False,
            batch_size=8
        )
        for i, result in enumerate(results):
            scores_map = dict(zip(result["labels"], result["scores"]))
            alliance   = scores_map.get(NLI_LABELS[0], 0.0)
            conflit    = scores_map.get(NLI_LABELS[1], 0.0)
            score_nli  = alliance - conflit
            edge_key   = edge_keys_for_texts[i]
            if edge_key not in edge_scores:
                edge_scores[edge_key] = []
            edge_scores[edge_key].append(score_nli)
    # ────────────────────────────────────────────────────────────────────────

    # Classifier chaque arête via le score NLI moyen de toutes ses fenêtres
    for (u, v), scores in edge_scores.items():
        if G.has_edge(u, v):
            score_moyen         = sum(scores) / len(scores)
            G[u][v]['relation'] = classify_relation(score_moyen)

    # Valeur par défaut pour les arêtes sans données NLI
    for u, v in G.edges():
        if 'relation' not in G[u][v]:
            G[u][v]['relation'] = 'neutre'

    # Suppression des arêtes sous le seuil de poids
    edges_to_remove = [
        (u, v) for u, v, data in G.edges(data=True)
        if data['weight'] < MIN_WEIGHT_THRESHOLD
    ]
    G.remove_edges_from(edges_to_remove)

    # Attribut 'names' sur chaque nœud
    for canon, variants in node_variants.items():
        if not G.has_node(canon): G.add_node(canon)
        variants.add(canon)
        G.nodes[canon]["names"] = ";".join(sorted(variants))

    # Suppression des nœuds isolés non-vitaux
    for node in list(nx.isolates(G)):
        if node not in vital_chars:
            G.remove_node(node)

    return G

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Graphe de personnages — DL Only (Zero-Shot NLI)")
    parser.add_argument("--LP",      default="outputs/LP_final.txt",          type=Path)
    parser.add_argument("--corpus",  default="corpus_asimov_leaderboard",     type=Path)
    parser.add_argument("-o", "--output", default="DL_only.csv",              type=Path)
    args = parser.parse_args()

    print("--- Génération DL_only (Zero-Shot NLI pur) ---")

    if not args.LP.exists():
        print("Erreur : Fichier LP introuvable. Avez-vous exécuté listLP.py ?")
        return

    corpus_path = normalize_corpus_path(args.corpus)
    if corpus_path != args.corpus:
        print(f"[INFO] Chemin corpus converti pour WSL : {args.corpus} -> {corpus_path}")
    if not corpus_path.exists():
        print(f"Erreur : Dossier corpus introuvable : {corpus_path}")
        return

    lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]
    alias_map, vital_chars = build_hybrid_alias_map(lp_list, corpus_path)

    df_dict       = {"ID": [], "graphml": []}
    chapter_count = 0

    for folder_name, book_code in BOOK_CODES.items():
        folder_path = corpus_path / folder_name
        if not folder_path.exists(): continue

        files = sorted(
            folder_path.glob("chapter_*.txt.preprocessed"),
            key=lambda p: int(re.search(r'\d+', p.name).group())
        )
        print(f"Traitement {book_code}...")
        for fpath in files:
            file_num  = int(re.search(r'\d+', fpath.name).group())
            chap_num  = max(0, file_num - 1)
            chap_id   = f"{book_code}{chap_num}"
            text      = fpath.read_text(encoding="utf-8")
            # Pas de lexique à passer : signature allégée
            G         = build_graph_for_chapter(text, alias_map, vital_chars)
            graphml_str = "".join(nx.generate_graphml(G))
            df_dict["ID"].append(chap_id)
            df_dict["graphml"].append(graphml_str)
            chapter_count += 1

    if chapter_count == 0:
        print("Erreur : aucun chapitre trouvé.")
        return

    print("Post-traitement des relations...")
    df_dict = smooth_relations_globally(df_dict, min_chapters=4, min_confidence=0.50)

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv(args.output)
    print(f"Terminé : {args.output}")


if __name__ == "__main__":
    main()

    # python3 graph_submission_DL_only.py --LP outputs/LP_final.txt --corpus corpus_asimov_leaderboard -o DL_only.csv
    # python3 eval_relations.py --input DL_only.csv
