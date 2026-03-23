#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_relations.py — Visualisation des graphes avec relations pour/contre/neutre

Couleurs des arêtes :
  vert   (#2ecc71) = POUR
  rouge  (#e74c3c) = CONTRE
  gris   (#95a5a6) = NEUTRE

Usage :
  # Un seul chapitre
  python3 visualize_relations.py my_submission_relation.csv -c paf0

  # Livre entier fusionné
  python3 visualize_relations.py my_submission_relation.csv -b lca

  # Tous les chapitres d'un livre (un fichier par chapitre)
  python3 visualize_relations.py my_submission_relation.csv -b paf --all-chapters

  # Changer le fichier de sortie
  python3 visualize_relations.py my_submission_relation.csv -c lca3 -o lca3.png
"""

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')   # Pas besoin d'interface graphique (compatible WSL)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import sys
import re
from pathlib import Path

# =============================================================================
# COULEURS
# =============================================================================
RELATION_COLORS = {
    "pour":    "#2ecc71",   # vert
    "contre":  "#e74c3c",   # rouge
    "neutre":  "#bdc3c7",   # gris clair
}
NODE_COLOR   = "#3498db"    # bleu
NODE_OUTLINE = "#2980b9"

# =============================================================================
# CHARGEMENT
# =============================================================================

def load_graph_from_csv(csv_path, chapter_id=None, merge_book=None):
    """
    Charge le graphe depuis le CSV avec l'attribut relation sur les arêtes.
    - chapter_id  : charge un seul chapitre (ex: 'paf0')
    - merge_book  : fusionne tous les chapitres d'un livre (ex: 'lca')
    """
    try:
        df = pd.read_csv(csv_path, index_col="ID")
    except Exception as e:
        print(f"Erreur de lecture CSV : {e}")
        sys.exit(1)

    if chapter_id:
        if chapter_id not in df.index:
            print(f"Erreur : L'ID '{chapter_id}' n'existe pas.")
            print("IDs disponibles :", list(df.index))
            sys.exit(1)
        G = nx.parse_graphml(df.loc[chapter_id, "graphml"])
        print(f"Graphe chargé : {chapter_id} ({len(G.nodes)} nœuds, {len(G.edges)} arêtes)")
        return G

    elif merge_book:
        print(f"Fusion des graphes pour le livre : {merge_book}...")
        G_final = nx.Graph()
        book_rows = df[df.index.str.startswith(merge_book)]

        if book_rows.empty:
            print(f"Aucun chapitre trouvé pour '{merge_book}'.")
            sys.exit(1)

        for chap_id, row in book_rows.iterrows():
            g_chap = nx.parse_graphml(row["graphml"])

            for u, v, data in g_chap.edges(data=True):
                weight   = float(data.get('weight', 1))
                relation = data.get('relation', 'neutre')

                if G_final.has_edge(u, v):
                    G_final[u][v]['weight'] += weight
                    # La relation est celle qui domine (mise à jour si poids plus élevé)
                    if weight > G_final[u][v].get('_best_weight', 0):
                        G_final[u][v]['relation']      = relation
                        G_final[u][v]['_best_weight']  = weight
                else:
                    G_final.add_edge(u, v, weight=weight, relation=relation,
                                     _best_weight=weight)

            for node in g_chap.nodes():
                if not G_final.has_node(node):
                    G_final.add_node(node)

        print(f"Graphe fusionné : {len(G_final.nodes)} nœuds, {len(G_final.edges)} arêtes")
        return G_final

    else:
        print("Erreur : Spécifiez --chapter ou --book")
        sys.exit(1)


# =============================================================================
# DESSIN
# =============================================================================

def draw_relation_graph(G, output_file, title, min_weight=1):
    """Dessine le graphe avec arêtes colorées par type de relation."""

    # Filtrer les arêtes trop légères si demandé
    if min_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if float(d.get('weight', 1)) < min_weight]
        G = G.copy()
        G.remove_edges_from(edges_to_remove)

    if len(G.nodes) == 0:
        print("Graphe vide après filtrage, rien à afficher.")
        return

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Layout
    if len(G.nodes) <= 10:
        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    elif len(G.nodes) <= 25:
        pos = nx.spring_layout(G, k=0.8, iterations=60, seed=42)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Tailles des nœuds (proportionnelles au degré)
    degrees    = dict(G.degree())
    node_sizes = [max(degrees[n] * 120 + 200, 300) for n in G.nodes()]

    # Arêtes groupées par relation
    edge_groups = {"pour": [], "contre": [], "neutre": []}
    edge_widths  = {"pour": [], "contre": [], "neutre": []}

    all_weights = [float(G[u][v].get('weight', 1)) for u, v in G.edges()]
    max_w = max(all_weights) if all_weights else 1

    for u, v, data in G.edges(data=True):
        rel    = data.get('relation', 'neutre')
        weight = float(data.get('weight', 1))
        width  = (weight / max_w) * 6 + 0.8

        if rel not in edge_groups:
            rel = 'neutre'
        edge_groups[rel].append((u, v))
        edge_widths[rel].append(width)

    # Dessin des arêtes par groupe
    for rel, edges in edge_groups.items():
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=edge_widths[rel],
            edge_color=RELATION_COLORS[rel],
            alpha=0.75,
            ax=ax
        )

    # Dessin des nœuds
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=NODE_COLOR,
        edgecolors=NODE_OUTLINE,
        linewidths=1.5,
        alpha=0.95,
        ax=ax
    )

    # Labels
    label_pos = {n: (x, y + 0.04) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos,
        font_size=8,
        font_weight="bold",
        font_color="white",
        ax=ax,
        bbox=dict(facecolor="#16213e", alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2')
    )

    # Légende
    legend_patches = [
        mpatches.Patch(color=RELATION_COLORS["pour"],   label=f"POUR ({len(edge_groups['pour'])} arêtes)"),
        mpatches.Patch(color=RELATION_COLORS["contre"], label=f"CONTRE ({len(edge_groups['contre'])} arêtes)"),
        mpatches.Patch(color=RELATION_COLORS["neutre"], label=f"NEUTRE ({len(edge_groups['neutre'])} arêtes)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=10,
        framealpha=0.85,
        facecolor="#16213e",
        edgecolor="#e94560",
        labelcolor="white"
    )

    # Statistiques
    stats_text = (f"Nœuds : {len(G.nodes)}  |  "
                  f"Arêtes : {len(G.edges)}  |  "
                  f"Pour : {len(edge_groups['pour'])}  "
                  f"Contre : {len(edge_groups['contre'])}  "
                  f"Neutre : {len(edge_groups['neutre'])}")
    ax.text(0.5, 0.01, stats_text,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color="#95a5a6")

    ax.set_title(title, fontsize=18, fontweight="bold", color="white", pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Image sauvegardée : {output_file}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualisation des graphes de personnages avec relations"
    )
    parser.add_argument("csv_file", type=str,
                        help="Fichier CSV (ex: my_submission_relation.csv)")
    parser.add_argument("-c", "--chapter", type=str,
                        help="ID du chapitre (ex: paf0, lca3)")
    parser.add_argument("-b", "--book", type=str,
                        help="Code livre pour fusionner (ex: paf, lca)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Fichier image de sortie (défaut: auto)")
    parser.add_argument("--all-chapters", action="store_true",
                        help="Générer une image par chapitre du livre (utiliser avec -b)")
    parser.add_argument("--min-weight", type=int, default=1,
                        help="Poids minimum pour afficher une arête (défaut: 1)")

    args = parser.parse_args()

    # Mode : tous les chapitres d'un livre
    if args.all_chapters and args.book:
        try:
            df = pd.read_csv(args.csv_file, index_col="ID")
        except Exception as e:
            print(f"Erreur : {e}"); sys.exit(1)

        book_rows = df[df.index.str.startswith(args.book)]
        if book_rows.empty:
            print(f"Aucun chapitre pour '{args.book}'"); sys.exit(1)

        out_dir = Path(f"graphes_{args.book}")
        out_dir.mkdir(exist_ok=True)

        for chap_id in book_rows.index:
            G     = load_graph_from_csv(args.csv_file, chapter_id=chap_id)
            out   = out_dir / f"{chap_id}.png"
            title = f"Réseau de Personnages — Chapitre {chap_id}"
            draw_relation_graph(G, str(out), title, min_weight=args.min_weight)

        print(f"\n✅ {len(book_rows)} images générées dans ./{out_dir}/")
        return

    # Mode : un seul chapitre
    if args.chapter:
        G     = load_graph_from_csv(args.csv_file, chapter_id=args.chapter)
        out   = args.output or f"graphe_{args.chapter}.png"
        title = f"Réseau de Personnages — Chapitre {args.chapter}"
        draw_relation_graph(G, out, title, min_weight=args.min_weight)
        return

    # Mode : livre entier fusionné
    if args.book:
        G     = load_graph_from_csv(args.csv_file, merge_book=args.book)
        out   = args.output or f"graphe_{args.book}_global.png"
        title = f"Réseau Global de Personnages — {args.book.upper()}"
        draw_relation_graph(G, out, title, min_weight=args.min_weight)
        return

    print("Erreur : Spécifiez -c (chapitre) ou -b (livre). Voir --help")


if __name__ == "__main__":
    main()
