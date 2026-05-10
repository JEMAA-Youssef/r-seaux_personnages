import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io

#Charger le CSV
df = pd.read_csv("my_submission.csv")
print(f"J'ai chargé {len(df)} chapitres au total.")

target_id = "paf5" 

Récupérer la chaîne XML GraphML correspondante
try:
    xml_content = df.loc[df['ID'] == target_id, 'graphml'].values[0]
except IndexError:
    print(f"ID {target_id} introuvable !")
    exit()

#Parser le GraphML
# On utilise io.BytesIO car networkx attend un "fichier"
G = nx.read_graphml(io.BytesIO(xml_content.encode('utf-8')))

# 4. Afficher les infos statistiques
print(f"--- Vérification : {target_id} ---")
print(f"Nombre de nœuds : {G.number_of_nodes()}")
print(f"Nombre d'arêtes : {G.number_of_edges()}")
print("\nAttributs 'names' (pour Kaggle) :")
for node, data in G.nodes(data=True):
    print(f" - {node} : {data.get('names', 'PAS DE NAMES !')}")

# Dessiner le graphe
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5) # k écarte les nœuds
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f"Visualisation de {target_id}")
output_filename = f"graph_{target_id}.png"
plt.savefig(output_filename)
print(f"\nImage sauvegardée sous : {output_filename}")
print("Vous pouvez maintenant ouvrir ce fichier image pour voir le graphe.")