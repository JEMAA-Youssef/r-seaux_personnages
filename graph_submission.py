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
    "Ciel", "Dieu", "Seigneur", "Quarantecinq", "Jenarr", "Leggen","Torres"
}

TITLES_TO_STRIP = {"Dr", "Docteur", "Maire", "Commissaire", "Mme", "M.", "Maître", "Sire", "Empereur", "R.", "Robot", "Général"}

# OUTILS

def normalize_name(name):
    """Normalise la casse."""
    parts = name.title().split() #mettre en majuscule la première lettre de chaque mot et découp la phrase en liste
    fixed_parts = []
    for p in parts:
        if re.match(r'^(Ier|Ii|Iii|Iv|V|Vi|Vii|Ix|X)$', p, re.IGNORECASE): #forcer la maj
            if p.lower() == 'ier': fixed_parts.append('Ier')
            else: fixed_parts.append(p.upper())
        elif p.lower() in ["de", "la", "von", "van", "of"]:#force le min
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
    
    #Scan du corpus pour les fréquences
    full_text = ""
    for folder_name in BOOK_CODES:
        folder_path = corpus_dir / folder_name
        if folder_path.exists():
            for f in folder_path.glob("*.txt.preprocessed"):
                full_text += f.read_text(encoding="utf-8") + "\n"
    
    normalized_lp = {normalize_name(n) for n in lp_list if n not in GRAPH_BLACKLIST}
    
    freqs = Counter()
    for name in normalized_lp:
        count = len(re.findall(r'\b' + re.escape(name) + r'\b', full_text, re.IGNORECASE))#calcul de fréquence 
        freqs[name] = count

    sorted_candidates = sorted(normalized_lp, key=lambda x: (freqs[x], len(x)), reverse=True)
    alias_map = {}
    
    #automatisation
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

    # 3. manuelle overrides (score)
    print("Application des correctifs manuels...")
    for short, target in CRITICAL_ALIASES.items():
        norm_short = normalize_name(short)
        norm_target = normalize_name(target)
        
        # On force le mapping
        alias_map[norm_short] = norm_target
        # On s'assure que la cible est bien une clé canonique (pointe vers elle-même)
        alias_map[norm_target] = norm_target

    #identification des Top 20
    final_counts = Counter()
    for name, count in freqs.items():
        # Attention : on utilise la map mise à jour
        if name in alias_map:
            canon = alias_map[name]
            final_counts[canon] += count
    
    # On sauve les 20 plus fréquents même s'ils sont isolés
    vital_chars = {name for name, _ in final_counts.most_common(20)}
    
    return alias_map, vital_chars


# MOTEUR DE GRAPHE

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

def build_graph_for_chapter(text, alias_map, vital_chars):
    G = nx.Graph()
    entities = get_entities_positions(text, alias_map)
    node_variants = {} 
    
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
                
            if G.has_edge(curr_canon, next_canon):
                G[curr_canon][next_canon]['weight'] += 1
            else:
                G.add_edge(curr_canon, next_canon, weight=1)

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LP", default="outputs/LP_final.txt", type=Path)
    parser.add_argument("--corpus", default="corpus_asimov_leaderboard", type=Path)
    parser.add_argument("-o", "--output", default="my_submission.csv", type=Path)
    args = parser.parse_args()

    print("--- Génération HYBRIDE (V5) ---")
    
    if not args.LP.exists(): return
    lp_list = [l.strip() for l in args.LP.open("r", encoding="utf-8") if l.strip()]
    
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
            G = build_graph_for_chapter(text, alias_map, vital_chars)
            
            graphml_str = "".join(nx.generate_graphml(G))
            df_dict["ID"].append(chap_id)
            df_dict["graphml"].append(graphml_str)

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv(args.output)
    print(f"Terminé : {args.output}")

if __name__ == "__main__":
    main()