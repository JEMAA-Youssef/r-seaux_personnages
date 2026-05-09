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
from transformers import pipeline
import torch

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
    "Rashelle": "Rashelle of Wye", "Wye": "Rashelle of Wye",
    "Mannix IV": "Mannix IV Kan", 
    "Mannix": "Mannix IV Kan"
}

# Mots à ignorer (Bruit)
GRAPH_BLACKLIST = {
    "Jésus", "Shakespeare", "Heisenberg", "Churchill", "Ahab", 
    "Job", "Naboth", "Jéhu", "Jéhoram", "Noé", "Moïse", 
    "Anciens", "Médiévaliste", "Médiévalistes", "Galactica", "Encyclopaedia",
    "Ciel", "Dieu", "Seigneur", "Quarantecinq", "Jenarr", "Leggen","Nord"
}

TITLES_TO_STRIP = {"Dr", "Docteur", "Maire", "Commissaire", "Mme", "M.", "Maître", "Sire", "Empereur", "R.", "Robot", "Général"}

# Chemin vers les ressources de sentiment
_RESOURCES_DIR = Path(__file__).parent / "resources"

# =============================================================================
# ANALYSE DE SENTIMENT AVEC NÉGATION
# =============================================================================

def analyze_window_zeroshot(window_text: str, classifier) -> Tuple[int, int]:
    """
    Analyse la relation dans une fenêtre de texte via Zero-Shot Classification.
    Retourne (pos_score, neg_score).
    """
    safe_text = window_text[:1500] 
    
    if not safe_text.strip():
        return 0, 0

    # Voici le "Prompt" caché. On donne à l'IA les 3 seuls choix possibles.
    candidate_labels = ["alliés ou amis", "ennemis ou conflit", "interaction neutre"]

    try:
        # hypothesis_template aide l'IA à formuler sa réflexion en français
        result = classifier(
            safe_text, 
            candidate_labels=candidate_labels,
            hypothesis_template="Dans ce texte, la relation entre les personnages est {}."
        )
        
        # L'IA classe les labels par ordre de probabilité. On prend le 1er (le plus probable)
        top_label = result['labels'][0]
        
        pos_score, neg_score = 0, 0
        
        if top_label == "ennemis ou conflit":
            neg_score = 1
        elif top_label == "alliés ou amis":
            pos_score = 1
        # Si c'est "interaction neutre", pos et neg restent à 0
            
        return pos_score, neg_score
        
    except Exception as e:
        print(f"Erreur Zero-Shot sur la fenêtre : {e}")
        return 0, 0


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


def normalize_corpus_path(path: Path) -> Path:
    """
    Convertit un chemin UNC WSL (//wsl.localhost/<distro>/...)
    en chemin Linux (/...) quand le script est execute sous WSL.
    """
    raw = str(path)
    match = re.match(r"^//wsl(?:\.localhost|\$)/[^/]+(/.*)$", raw, re.IGNORECASE)
    if match:
        return Path(match.group(1))
    return path

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
    # Enregistrer les namespaces pour éviter les préfixes ns0: dans la sortie
    ET.register_namespace('', _NS)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

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

def build_graph_for_chapter(text, alias_map, vital_chars, relation_classifier):
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
            pos, neg = analyze_window_zeroshot(window_text, relation_classifier)
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

def debug_pair(text, alias_map, sentiment_analyzer, charA, charB):
    """
    Affiche l'évaluation CamemBERT des fenêtres entre charA et charB.
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

            pos, neg = analyze_window_zeroshot(window_text, relation_classifier)
            print(f"  Fenêtre {windows_found} [{start_w}-{end_w}] : pos={pos} neg={neg} → {classify_relation(pos,neg)}")
            print(f"    Texte : {window_text[:300]}...") # On affiche un peu plus de texte
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

    if not args.LP.exists():
        print("Erreur: Fichier LP introuvable. Avez-vous exécuté listLP.py ?")
        return

    corpus_path = normalize_corpus_path(args.corpus)
    if corpus_path != args.corpus:
        print(f"[INFO] Chemin corpus converti pour WSL : {args.corpus} -> {corpus_path}")

    if not corpus_path.exists():
        print(f"Erreur: Dossier corpus introuvable : {corpus_path}")
        print("Conseil: sous Linux/WSL, utilisez un chemin de type /home/... et non //wsl.localhost/...")
        return

    lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]

    print("🧠 Chargement du modèle Zero-Shot NLI (cela peut prendre quelques secondes)...")
    
    # On utilise un modèle multilingue très robuste (mDeBERTa) parfait pour le français
    relation_classifier = pipeline(
        task="zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", # <-- LE NOUVEAU MODÈLE
        device=-1 
    )

    # ── Mode DEBUG ────────────────────────────────────────────────────────────
    if args.debug_pair:
        chap_id, perso_a, perso_b = args.debug_pair
        alias_map, vital_chars = build_hybrid_alias_map(lp_list, corpus_path)

        book_code = chap_id[:3]          # ex: 'paf'
        chap_num  = int(chap_id[3:]) + 1  # ex: 0 -> chapter_1
        folder_name = {v: k for k, v in BOOK_CODES.items()}.get(book_code)
        if not folder_name:
            print(f"Code livre inconnu: {book_code}"); return
        fpath = corpus_path / folder_name / f"chapter_{chap_num}.txt.preprocessed"
        if not fpath.exists():
            print(f"Fichier introuvable: {fpath}"); return

        text = fpath.read_text(encoding="utf-8")
        print(f"\n=== DEBUG relation : '{perso_a}' ↔ '{perso_b}' (chapitre {chap_id}) ===")
        debug_pair(text, alias_map, relation_classifier, perso_a, perso_b)
        return
    # ─────────────────────────────────────────────────────────────────────────

    # Construction Hybride (Auto + Manuel)
    alias_map, vital_chars = build_hybrid_alias_map(lp_list, corpus_path)

    df_dict = {"ID": [], "graphml": []}
    chapter_count = 0
    
    for folder_name, book_code in BOOK_CODES.items():
        folder_path = corpus_path / folder_name
        if not folder_path.exists(): continue
            
        files = sorted(folder_path.glob("chapter_*.txt.preprocessed"), 
                       key=lambda p: int(re.search(r'\d+', p.name).group()))
        
        print(f"Traitement {book_code}...")
        for fpath in files:
            file_num = int(re.search(r'\d+', fpath.name).group())
            chap_num = max(0, file_num - 1)
            chap_id = f"{book_code}{chap_num}"
            
            text = fpath.read_text(encoding="utf-8")
            G = build_graph_for_chapter(text, alias_map, vital_chars, relation_classifier)
            graphml_str = "".join(nx.generate_graphml(G))
            df_dict["ID"].append(chap_id)
            df_dict["graphml"].append(graphml_str)
            chapter_count += 1

    if chapter_count == 0:
        print("Erreur: aucun chapitre trouve. Verifiez --corpus et la structure des dossiers.")
        return

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



    #python3 graph_submission.py --LP outputs/LP_final.txt --corpus corpus_asimov_leaderboard -o my_submission_v5.csv