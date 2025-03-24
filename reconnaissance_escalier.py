import json
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np

def load_labels(json_path):
    """
    Charge les étiquettes à partir du fichier JSON et retourne un dictionnaire avec les points de chaque label.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    labels = {'escalier': [], 'surfaceH1': [], 'surfaceV1': []}
    
    # Parcourir les formes dans 'shapes' et associer les points aux labels correspondants
    for shape in data['shapes']:
        label = shape['label']
        if label in labels:
            labels[label].append(np.array(shape['points']))
    
    return labels

def plot_labels(image_path, json_path):
    """
    Affiche l'image avec les labels et les points de polygones dessinés.
    """
    # Charger l'image
    img = mplimg.imread(image_path)
    
    # Charger les labels
    labels = load_labels(json_path)
    
    # Affichage de l'image
    plt.imshow(img)
    
    # Dessiner les polygones pour chaque label
    for label, points_list in labels.items():
        for points in points_list:
            # Convertir les points en une forme compatible pour tracer
            poly = plt.Polygon(points, fill=None, edgecolor='r', linewidth=2)
            plt.gca().add_patch(poly)
    
    plt.axis('off')  # Masquer les axes
    plt.show()

# Exemple d'utilisation
image_path = "images/6.jpg"
json_path = "images/6.json"
plot_labels(image_path, json_path)