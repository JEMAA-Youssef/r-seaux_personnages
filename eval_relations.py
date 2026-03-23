#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_relations.py — Évaluation automatique des relations personnages

Stratégie : évaluation sans annotation manuelle, basée sur 3 critères :
  1. Cohérence inter-chapitres : une paire (A,B) doit garder la même
     relation dans la majorité des chapitres où elle apparaît.
  2. Couverture : % d'arêtes classées "pour" ou "contre" (non-neutre).
  3. Corrélation poids/relation : les paires fréquentes (poids élevé)
     devraient avoir une relation marquée (non-neutre).

Usage :
    python3 eval_relations.py --input my_submission_relation.csv
    python3 eval_relations.py --input my_submission_relation.csv --detail
    python3 eval_relations.py --input my_submission_relation.csv --export eval_report.csv
"""

import argparse
import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

NS = {'g': 'http://graphml.graphdrawing.org/xmlns'}


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_edges(graphml_str):
    """Extrait les arêtes (src, tgt, weight, relation) d'un GraphML."""
    root = ET.fromstring(graphml_str)
    keys = {k.get('id'): k.get('attr.name') for k in root.findall('g:key', NS)}
    edges = []
    for edge in root.findall('.//g:edge', NS):
        src = edge.get('source')
        tgt = edge.get('target')
        data = {keys[d.get('key')]: d.text for d in edge.findall('g:data', NS)}
        weight   = int(data.get('weight', 0))
        relation = data.get('relation', 'neutre')
        edges.append((src, tgt, weight, relation))
    return edges


def load_all_edges(csv_path):
    """
    Charge toutes les arêtes de tous les chapitres.
    Retourne une liste de dicts avec les champs :
      chapitre, livre, src, tgt, paire, weight, relation
    """
    df = pd.read_csv(csv_path, index_col='ID')
    rows = []
    for chap_id, row in df.iterrows():
        livre = ''.join(filter(str.isalpha, chap_id))  # 'paf' ou 'lca'
        try:
            edges = extract_edges(row['graphml'])
        except Exception:
            continue
        for src, tgt, weight, relation in edges:
            paire = tuple(sorted([src, tgt]))
            rows.append({
                'chapitre': chap_id,
                'livre':    livre,
                'src':      src,
                'tgt':      tgt,
                'paire':    paire,
                'weight':   weight,
                'relation': relation,
            })
    return pd.DataFrame(rows)


# =============================================================================
# MÉTRIQUES
# =============================================================================

def metric_coherence(df):
    """
    Cohérence inter-chapitres par paire.

    Pour chaque paire (A,B), on regarde la distribution de la relation
    à travers tous les chapitres où elle apparaît.
    Cohérence = occurrence de la relation majoritaire / total apparitions.

    Retourne un DataFrame trié par cohérence croissante (les pires d'abord).
    """
    records = []
    for paire, group in df.groupby('paire'):
        counts = Counter(group['relation'])
        total  = len(group)
        top_rel, top_freq = counts.most_common(1)[0]
        coherence = top_freq / total
        records.append({
            'paire':             f"{paire[0]} / {paire[1]}",
            'apparitions':       total,
            'relation_dominante': top_rel,
            'coherence_%':       round(coherence * 100, 1),
            'distribution':      dict(counts),
        })
    return pd.DataFrame(records).sort_values('coherence_%')


def metric_coverage(df):
    """
    Taux de couverture : % d'arêtes ayant une relation marquée (non-neutre).
    Ventilé par livre et global.
    """
    results = {}
    for livre in ['paf', 'lca', 'global']:
        sub = df if livre == 'global' else df[df['livre'] == livre]
        if len(sub) == 0:
            continue
        total    = len(sub)
        non_neutre = len(sub[sub['relation'] != 'neutre'])
        pour     = len(sub[sub['relation'] == 'pour'])
        contre   = len(sub[sub['relation'] == 'contre'])
        neutre_  = len(sub[sub['relation'] == 'neutre'])
        results[livre] = {
            'total_arêtes':   total,
            'pour':           pour,
            'contre':         contre,
            'neutre':         neutre_,
            'couverture_%':   round(non_neutre / total * 100, 1),
            'pour_%':         round(pour       / total * 100, 1),
            'contre_%':       round(contre     / total * 100, 1),
            'neutre_%':       round(neutre_    / total * 100, 1),
        }
    return results


def metric_weight_vs_relation(df):
    """
    Corrélation poids/relation :
    Les paires très fréquentes (poids élevé) devraient avoir
    une relation marquée. On vérifie si les arêtes "neutre" ont
    tendance à avoir un poids plus faible.
    """
    avg_weight = df.groupby('relation')['weight'].mean().round(1).to_dict()
    return avg_weight


def metric_top_pairs(df, n=20):
    """
    Top N paires les plus fréquentes avec leur relation agrégée.
    La relation agrégée = majorité sur tous les chapitres.
    """
    records = []
    for paire, group in df.groupby('paire'):
        counts  = Counter(group['relation'])
        top_rel = counts.most_common(1)[0][0]
        total_w = group['weight'].sum()
        records.append({
            'personnage_A':   paire[0],
            'personnage_B':   paire[1],
            'relation':       top_rel,
            'poids_total':    total_w,
            'nb_chapitres':   len(group),
            'distribution':   dict(counts),
        })
    return (pd.DataFrame(records)
              .sort_values('poids_total', ascending=False)
              .head(n))


def metric_inconsistent_pairs(df, seuil_coherence=0.6, min_apparitions=3):
    """
    Paires incohérentes : apparaissent souvent mais changent de relation.
    Ce sont les candidats à corriger dans le lexique.
    """
    coh = metric_coherence(df)
    inconsistent = coh[
        (coh['coherence_%'] < seuil_coherence * 100) &
        (coh['apparitions'] >= min_apparitions)
    ]
    return inconsistent


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_separator(titre):
    print(f"\n{'='*70}")
    print(f"  {titre}")
    print('='*70)


def print_coverage(coverage):
    print_separator("1. COUVERTURE DES RELATIONS")
    for livre, stats in coverage.items():
        print(f"\n  [{livre.upper()}]")
        print(f"    Total arêtes  : {stats['total_arêtes']}")
        print(f"    ✅ pour       : {stats['pour']:>4d}  ({stats['pour_%']:>5.1f}%)")
        print(f"    ❌ contre     : {stats['contre']:>4d}  ({stats['contre_%']:>5.1f}%)")
        print(f"    ⚪ neutre     : {stats['neutre']:>4d}  ({stats['neutre_%']:>5.1f}%)")
        print(f"    → Couverture  : {stats['couverture_%']}% d'arêtes classées (non-neutre)")


def print_weight_relation(avg):
    print_separator("2. POIDS MOYEN PAR TYPE DE RELATION")
    print("  (Un poids élevé sur 'neutre' = signal faible du lexique)\n")
    for rel in ['pour', 'contre', 'neutre']:
        bar = '█' * int(avg.get(rel, 0) / 3)
        print(f"  {rel:7s} : {avg.get(rel, 0):>6.1f}  {bar}")


def print_top_pairs(top_df):
    print_separator("3. TOP 20 PAIRES LES PLUS FRÉQUENTES")
    print(f"  {'Relation':8s} {'Poids':>6s} {'Chap':>5s}  Paire")
    print(f"  {'-'*8} {'-'*6} {'-'*5}  {'-'*45}")
    for _, row in top_df.iterrows():
        emoji = {'pour': '✅', 'contre': '❌', 'neutre': '⚪'}.get(row['relation'], '?')
        print(f"  {emoji} {row['relation']:6s} {row['poids_total']:>6d} {row['nb_chapitres']:>5d}  "
              f"{row['personnage_A']} / {row['personnage_B']}")
        dist = row['distribution']
        if len(dist) > 1:
            detail = "  ".join(f"{r}:{c}" for r, c in dist.items())
            print(f"           {'':>6s} {'':>5s}  ↳ [{detail}]")


def print_inconsistent(incons_df):
    print_separator("4. PAIRES INCOHÉRENTES (à corriger en priorité)")
    if len(incons_df) == 0:
        print("  ✅ Aucune paire incohérente détectée (seuil 60%, min 3 apparitions)")
        return
    print(f"  {'Cohérence':>10s} {'App':>4s} {'Dominant':8s}  Paire")
    print(f"  {'-'*10} {'-'*4} {'-'*8}  {'-'*45}")
    for _, row in incons_df.iterrows():
        print(f"  {row['coherence_%']:>9.1f}% {row['apparitions']:>4d} "
              f"{row['relation_dominante']:8s}  {row['paire']}")
        print(f"  {'':>10s} {'':>4s} {'':>8s}  ↳ {row['distribution']}")


def print_score_global(coverage, avg_weight, incons_count):
    print_separator("5. SCORE GLOBAL DE QUALITÉ")

    couv = coverage.get('global', {}).get('couverture_%', 0)

    # Critère 1 : couverture (objectif > 40%)
    score_couv = min(couv / 40 * 40, 40)

    # Critère 2 : poids neutre vs marqué
    w_neutre = avg_weight.get('neutre', 0)
    w_marque = (avg_weight.get('pour', 0) + avg_weight.get('contre', 0)) / 2
    if w_marque > 0:
        ratio = min(w_marque / max(w_neutre, 1), 2)
        score_weight = min(ratio / 2 * 30, 30)
    else:
        score_weight = 0

    # Critère 3 : cohérence (moins d'iincohérences = mieux, max 30pts)
    score_coh = max(30 - incons_count * 3, 0)

    total = round(score_couv + score_weight + score_coh)

    print(f"\n  Couverture    ({couv:.1f}% / obj 40%)   : {score_couv:.0f}/40 pts")
    print(f"  Poids/Rel     (marqué vs neutre)    : {score_weight:.0f}/30 pts")
    print(f"  Cohérence     ({incons_count} paires incoh.)   : {score_coh:.0f}/30 pts")
    print(f"\n  ★ SCORE TOTAL : {total}/100")

    if total >= 75:
        verdict = "✅ Excellent — lexique bien calibré"
    elif total >= 50:
        verdict = "⚠️  Correct — affiner le lexique sur les paires incohérentes"
    else:
        verdict = "❌ Faible — le lexique manque de couverture ou de précision"
    print(f"  {verdict}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='my_submission_relation.csv', type=Path)
    parser.add_argument('--detail', action='store_true',
                        help='Afficher toutes les paires (pas seulement le top 20)')
    parser.add_argument('--export', default=None, type=Path,
                        help='Exporter le rapport détaillé en CSV')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Fichier introuvable : {args.input}")
        return

    print(f"Chargement de {args.input}...")
    df = load_all_edges(args.input)
    print(f"  {len(df)} arêtes chargées sur {df['chapitre'].nunique()} chapitres")

    # Calcul des métriques
    coverage     = metric_coverage(df)
    avg_weight   = metric_weight_vs_relation(df)
    top_pairs    = metric_top_pairs(df, n=None if args.detail else 20)
    incons       = metric_inconsistent_pairs(df)

    # Affichage
    print_coverage(coverage)
    print_weight_relation(avg_weight)
    print_top_pairs(top_pairs)
    print_inconsistent(incons)
    print_score_global(coverage, avg_weight, len(incons))

    # Export CSV
    if args.export:
        all_pairs = metric_top_pairs(df, n=10000)
        all_pairs.to_csv(args.export, index=False)
        print(f"Rapport exporté : {args.export}")


if __name__ == '__main__':
    main()
