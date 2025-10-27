#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_submission.py - Construit le fichier de soumission Kaggle.

Ce script réalise les tâches suivantes :
1. Charge la liste finale des personnages (LP_final.txt).
2. Résout les alias pour regrouper les noms (ex: "Hari" et "Hari Seldon").
3. Itère sur chaque chapitre des livres requis ("paf" et "lca").
4. Pour chaque chapitre, détecte les co-occurrences de personnages dans les mêmes
   phrases pour modéliser leurs interactions.
5. Construit un graphe networkx pour le chapitre.
6. Attribue les alias à chaque nœud du graphe dans l'attribut "names".
7. Exporte les résultats dans un fichier CSV au format requis par Kaggle.
"""

import argparse
import itertools
import re
from pathlib import Path

import networkx as nx
import pandas as pd

# --- Configuration des chemins et constantes ---
DEFAULT_CHARACTERS_PATH = "LP_final.txt"
DEFAULT_CORPUS_DIR = "corpus_asimov_leaderboard"
DEFAULT_OUTPUT_CSV = "my_submission.csv"

# Structure des livres attendue par Kaggle
BOOKS = [
    ("paf", 19),  # Prélude à Fondation, 19 chapitres (0 à 18)
    ("lca", 18),  # Les Cavernes d'Acier, 18 chapitres (0 à 17)
]


def load_character_list(file_path: Path) -> list[str]:
    """Charge la liste de personnages depuis un fichier."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Le fichier des personnages est introuvable : {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return sorted([line.strip() for line in f if line.strip()], key=len, reverse=True)


def resolve_aliases(characters: list[str]) -> dict[str, set[str]]:
    """
    Regroupe les noms de personnages en alias.

    Stratégie simple : un nom plus court est considéré comme un alias d'un nom
    plus long s'il en est une sous-partie.
    Exemple: "Hari" et "Seldon" deviennent des alias de "Hari Seldon".

    Returns:
        Un dictionnaire où la clé est le nom canonique (le plus long) et la
        valeur est un ensemble de tous ses alias.
    """
    alias_map = {}
    # La liste est triée par longueur (décroissante), donc on rencontre
    # toujours le nom le plus long en premier.
    for name in characters:
        is_alias = False
        for canonical_name in alias_map:
            # Si "Hari" est dans "Hari Seldon"
            if f" {name} " in f" {canonical_name} ":
                alias_map[canonical_name].add(name)
                is_alias = True
                break
        if not is_alias:
            alias_map[name] = {name}
    return alias_map


def find_characters_in_text(text: str, alias_map: dict[str, set[str]]) -> dict[str, str]:
    """
    Trouve tous les alias dans un texte et les mappe à leur nom canonique.

    Returns:
        Un dictionnaire où la clé est l'alias trouvé dans le texte et la valeur
        est le nom canonique correspondant.
    """
    found_map = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            # Utilise \b pour s'assurer de ne trouver que des mots complets
            if re.search(r'\b' + re.escape(alias) + r'\b', text):
                found_map[alias] = canonical
    return found_map


def main():
    """Fonction principale pour générer le fichier de soumission."""
    parser = argparse.ArgumentParser(description="Générer la soumission Kaggle.")
    parser.add_argument("--characters", type=Path, default=DEFAULT_CHARACTERS_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    # --- Étape 1: Charger les personnages et résoudre les alias ---
    print("1. Résolution des alias...")
    all_characters = load_character_list(args.characters)
    alias_map = resolve_aliases(all_characters)
    print(f"   -> {len(alias_map)} personnages canoniques identifiés.")

    submission_data = {"ID": [], "graphml": []}

    # --- NOUVEAU : Mapping entre les codes de livre et les noms de dossier ---
    book_folder_map = {
        "paf": "prelude_a_fondation",
        "lca": "les_cavernes_d_acier"
    }

    # --- Étape 2: Itérer sur chaque chapitre ---
    print("\n2. Construction des graphes par chapitre...")
    for book_code, num_chapters in BOOKS:
        for chapter_num in range(num_chapters):
            chapter_id = f"{book_code}{chapter_num}"

            # --- MODIFIÉ : Construction correcte du chemin du fichier ---
            book_folder = book_folder_map.get(book_code)
            if not book_folder:
                print(f"   - Attention : code de livre inconnu '{book_code}', ignoré.")
                continue

            # Les noms de fichiers sont 1-indexés (chapter_1), mais la boucle est 0-indexée.
            # On ajoute donc 1 à chapter_num.
            file_name = f"chapter_{chapter_num + 1}.txt.preprocessed"
            chapter_file = args.corpus / book_folder / file_name

            if not chapter_file.is_file():
                print(f"   - Attention : fichier {chapter_file} introuvable, ignoré.")
                continue

            print(f"   - Traitement de {chapter_id}...")

            # --- Le reste de la boucle est inchangé ---
            chapter_text = chapter_file.read_text(encoding="utf-8")
            sentences = re.split(r'[.!?]+', chapter_text)

            G = nx.Graph()

            for sentence in sentences:
                chars_in_sentence = set()
                found_aliases = find_characters_in_text(sentence, alias_map)

                for alias, canonical in found_aliases.items():
                    chars_in_sentence.add(canonical)

                if len(chars_in_sentence) >= 2:
                    for char1, char2 in itertools.combinations(sorted(list(chars_in_sentence)), 2):
                        G.add_edge(char1, char2)

            for node in G.nodes():
                if node in alias_map:
                    names_attribute = ";".join(sorted(list(alias_map[node])))
                    G.nodes[node]["names"] = names_attribute

            graphml_string = "".join(nx.generate_graphml(G))

            submission_data["ID"].append(chapter_id)
            submission_data["graphml"].append(graphml_string)

    # --- Étape 5: Création du fichier CSV final ---
    print("\n3. Création du fichier de soumission CSV...")
    df = pd.DataFrame(submission_data)
    df.set_index("ID", inplace=True)
    df.to_csv(args.output)
    print(f"   -> Soumission sauvegardée dans '{args.output}'.")
    print("\nOpération terminée.")


if __name__ == "__main__":
    main()