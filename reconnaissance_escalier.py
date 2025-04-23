import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
#      GÉNÉRATION MASQUES
# =========================

def generate_mask_from_polygons(polygons, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        poly_np = np.array(poly, np.int32)
        cv2.fillPoly(mask, [poly_np], color=1)
    return mask

def load_ground_truth_mask(json_path, image_shape, label_name='escalier'):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in data['shapes']:
        if shape['label'] == label_name:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], color=1)

    return mask

# =========================
#     MÉTRIQUES MANUELLES
# =========================

def calcul_metrique(gt_mask, pred_mask):
    gt = gt_mask.flatten()
    pred = pred_mask.flatten()

    TP = np.sum((gt == 1) & (pred == 1))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    rappel = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) != 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

    return {
        'Précision': precision,
        'Rappel': rappel,
        'F1-score': f1,
        'IoU': iou
    }

# =========================
#      AFFICHAGE UTILE
# =========================

def afficher_masques(image, gt_mask, pred_mask):
    """
    Affiche l'image originale avec superposition semi-transparente des masques de GT et prédiction.
    Vert : vérité terrain, Rouge : prédiction
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axs[0].imshow(image_rgb)
    axs[0].set_title("Image originale")
    axs[0].axis('off')

    # Superposition GT (vert transparent)
    gt_overlay = image_rgb.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    gt_overlay[gt_mask == 1] = 0.5 * gt_overlay[gt_mask == 1] + 0.5 * green
    axs[1].imshow(gt_overlay.astype(np.uint8))
    axs[1].set_title("Masque Vérité Terrain (vert)")
    axs[1].axis('off')

    # Superposition prédiction (rouge transparent)
    pred_overlay = image_rgb.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    pred_overlay[pred_mask == 1] = 0.5 * pred_overlay[pred_mask == 1] + 0.5 * red
    axs[2].imshow(pred_overlay.astype(np.uint8))
    axs[2].set_title("Masque de Prédiction (rouge)")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


# =========================
#     TRAITEMENT EN LOT
# =========================
def traiter_dossier(dossier_images, dossier_jsons):
    fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith('5.jpg')])

    for fichier_image in fichiers:
        nom_base = os.path.splitext(fichier_image)[0]
        chemin_image = os.path.join(dossier_images, fichier_image)
        chemin_json_gt = os.path.join(dossier_jsons, f"{nom_base}.json")
        chemin_json_pred = os.path.join("Annotations", f"{nom_base}.json")  # <- annotations prédictives

        if not os.path.exists(chemin_json_gt):
            print(f"[!] Annotation Vérité Terrain manquante pour : {fichier_image}")
            continue

        if not os.path.exists(chemin_json_pred):
            print(f"[!] Annotation de prédiction manquante pour : {fichier_image}")
            continue

        image = cv2.imread(chemin_image)

        gt_mask = load_ground_truth_mask(chemin_json_gt, image.shape)
        pred_mask = load_ground_truth_mask(chemin_json_pred, image.shape)

        scores = calcul_metrique(gt_mask, pred_mask)
        print(f"Résultats pour {fichier_image} :")
        for metrique, valeur in scores.items():
            print(f"  {metrique} : {valeur:.4f}")

        afficher_masques(image, gt_mask, pred_mask)


# =========================
#         MAIN
# =========================

if __name__ == "__main__":
    dossier_images = "Base_Validation"
    dossier_jsons = "json"
    traiter_dossier(dossier_images, dossier_jsons)
