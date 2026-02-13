#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_graph.py — Visualisation esthétique pour rapport/présentation
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import sys

# Si vous voulez un style plus "pro" (nécessite d'avoir une police compatible, sinon commentez)
# plt.style.use('seaborn-v0_8-whitegrid') 

def load_graph_from_csv(csv_path, chapter_id=None, merge_book=None):
    """
    Charge le graphe depuis le CSV.
    - Si chapter_id est donné (ex: 'lca1'), charge ce chapitre.
    - Si merge_book est donné (ex: 'lca'), fusionne tous les chapitres de ce livre.
    """
    try:
        df = pd.read_csv(csv_path, index_col="ID")
    except Exception as e:
        print(f"Erreur de lecture CSV : {e}")
        sys.exit(1)

    # Cas 1 : Un seul chapitre
    if chapter_id:
        if chapter_id not in df.index:
            print(f"Erreur : L'ID '{chapter_id}' n'existe pas dans le CSV.")
            print("IDs disponibles :", list(df.index)[:5], "...")
            sys.exit(1)
        
        graphml_str = df.loc[chapter_id, "graphml"]
        # parse_graphml attend des bytes ou une liste de lignes
        G = nx.parse_graphml(graphml_str)
        print(f"Graph chargé : {chapter_id} ({len(G.nodes)} nœuds, {len(G.edges)} arêtes)")
        return G

    # Cas 2 : Fusion d'un livre entier (ex: 'lca')
    elif merge_book:
        print(f"Fusion des graphes pour le livre : {merge_book}...")
        G_final = nx.Graph()
        
        # Filtre les IDs qui commencent par le code livre (ex: 'lca0', 'lca1'...)
        book_rows = df[df.index.str.startswith(merge_book)]
        
        if book_rows.empty:
            print(f"Aucun chapitre trouvé pour le code '{merge_book}'.")
            sys.exit(1)

        for chap_id, row in book_rows.iterrows():
            g_chap = nx.parse_graphml(row["graphml"])
            
            # Fusion intelligente des poids
            for u, v, data in g_chap.edges(data=True):
                weight = float(data.get('weight', 1))
                if G_final.has_edge(u, v):
                    G_final[u][v]['weight'] += weight
                else:
                    G_final.add_edge(u, v, weight=weight)
            
            # Ajout des nœuds (pour garder ceux isolés si besoin)
            for node in g_chap.nodes():
                if not G_final.has_node(node):
                    G_final.add_node(node)
        
        print(f"Graphe fusionné : {len(G_final.nodes)} nœuds, {len(G_final.edges)} arêtes")
        return G_final

    else:
        print("Erreur : Spécifiez un --chapter ou un --book")
        sys.exit(1)

def draw_pretty_graph(G, output_file, title):
    plt.figure(figsize=(12, 12)) # Grande image carrée
    
    # 1. Layout (Disposition)
    # k=0.5 espace les nœuds (plus k est grand, plus c'est espacé)
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
    
    # 2. Calcul des tailles
    # Degré (nombre de voisins) pour la taille des nœuds
    d = dict(G.degree)
    node_sizes = [v * 100 + 100 for v in d.values()] # Ajustez le multiplicateur
    
    # Poids pour l'épaisseur des liens
    weights = [float(G[u][v]['weight']) for u, v in G.edges()]
    # Normalisation pour que les traits ne soient pas trop gros
    max_w = max(weights) if weights else 1
    width = [(w / max_w) * 5 + 0.5 for w in weights] # Épaisseur entre 0.5 et 5.5

    # 3. Dessin
    # Les Arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=width, edge_color="red")
    
    # Les Nœuds
    # On peut utiliser une couleur par degré ou fixe
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#6495ED", alpha=0.9, edgecolors="white")
    
    # Les Labels (Noms)
    # On décale un peu le label pour qu'il ne soit pas SUR le point
    label_pos = {k: (v[0], v[1]+0.04) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight="bold", font_family="sans-serif", 
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor='none', pad=1))

    plt.title(title, fontsize=20, fontweight="bold")
    plt.axis('off') # Cache les axes X/Y
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Image sauvegardée : {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Dessiner le graphe")
    parser.add_argument("csv_file", type=str, help="Le fichier CSV de soumission")
    parser.add_argument("-c", "--chapter", type=str, help="ID du chapitre (ex: lca1)")
    parser.add_argument("-b", "--book", type=str, help="Code livre pour fusionner (ex: lca ou paf)")
    parser.add_argument("-o", "--output", type=str, default="graphe_output.png", help="Nom du fichier image")
    
    args = parser.parse_args()
    
    # Titre automatique
    if args.chapter:
        title = f"Réseau de Personnages - Chapitre {args.chapter}"
        G = load_graph_from_csv(args.csv_file, chapter_id=args.chapter)
    elif args.book:
        title = f"Réseau Global - {args.book.upper()}"
        G = load_graph_from_csv(args.csv_file, merge_book=args.book)
    else:
        print("Il faut choisir soit un chapitre (-c) soit un livre entier (-b)")
        return

    draw_pretty_graph(G, args.output, title)

if __name__ == "__main__":
    main()