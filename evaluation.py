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
#     MÉTRIQUES 
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
    Affiche l'image originale avec :
    1. l'image seule
    2. la vérité terrain (vert)
    3. la prédiction (rouge)
    4. la superposition des deux :
       - Vert = vérité terrain correcte
       - Rouge = prédiction incorrecte
       - Jaune = vrai positif (prédit + vrai)
    """
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axs[0].imshow(image_rgb)
    axs[0].set_title("Image originale")
    axs[0].axis('off')

    # Masque GT
    gt_overlay = image_rgb.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    gt_overlay[gt_mask == 1] = 0.5 * gt_overlay[gt_mask == 1] + 0.5 * green
    axs[1].imshow(gt_overlay.astype(np.uint8))
    axs[1].set_title("Masque Vérité Terrain (vert)")
    axs[1].axis('off')

    # Masque prédiction
    pred_overlay = image_rgb.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    pred_overlay[pred_mask == 1] = 0.5 * pred_overlay[pred_mask == 1] + 0.5 * red
    axs[2].imshow(pred_overlay.astype(np.uint8))
    axs[2].set_title("Masque Prédiction (rouge)")
    axs[2].axis('off')

    # Superposition combinée
    fusion = image_rgb.copy()
    jaune = np.array([255, 255, 0], dtype=np.uint8)  # TP = jaune
    vert = np.array([0, 255, 0], dtype=np.uint8)     # FN = vert seul
    rouge = np.array([255, 0, 0], dtype=np.uint8)    # FP = rouge seul

    tp = (gt_mask == 1) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask == 1)

    fusion[tp] = 0.5 * fusion[tp] + 0.5 * jaune
    fusion[fn] = 0.5 * fusion[fn] + 0.5 * vert
    fusion[fp] = 0.5 * fusion[fp] + 0.5 * rouge

    axs[3].imshow(fusion.astype(np.uint8))
    axs[3].set_title("Fusion : Vert=FN, Rouge=FP, Jaune=TP")
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()



# =========================
#   ÉVALUATION GLOBALE
# =========================

def evaluation_globale(dossier_images, dossier_jsons_gt, dossier_jsons_pred, seuil_iou=0.5):
    """
    Évalue globalement la détection sur un dossier complet :
    - Calcule le pourcentage d'escaliers correctement détectés (IoU > seuil).
    """

    fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith('.jpg')])

    total_images = 0
    escaliers_detectes = 0

    for fichier_image in fichiers:
        nom_base = os.path.splitext(fichier_image)[0]
        chemin_image = os.path.join(dossier_images, fichier_image)
        chemin_json_gt = os.path.join(dossier_jsons_gt, f"{nom_base}.json")     # Vérité terrain
        chemin_json_pred = os.path.join(dossier_jsons_pred, f"{nom_base}.json") # Prédiction

        if not os.path.exists(chemin_json_gt) or not os.path.exists(chemin_json_pred):
            print(f"[!] Fichier manquant pour {fichier_image}, saut de l'image.")
            continue

        image = cv2.imread(chemin_image)
        gt_mask = load_ground_truth_mask(chemin_json_gt, image.shape)
        pred_mask = load_ground_truth_mask(chemin_json_pred, image.shape)

        scores = calcul_metrique(gt_mask, pred_mask)
        print(f"Résultats pour {fichier_image} :")
        for metrique, valeur in scores.items():
            print(f"  {metrique} : {valeur:.4f}")

        #afficher_masques(image, gt_mask, pred_mask)
        iou = scores['IoU']

        total_images += 1
        if iou >= seuil_iou:
            escaliers_detectes += 1

    if total_images == 0:
        print("[!] Aucune image évaluée.")
        return

    pourcentage = (escaliers_detectes / total_images) * 100
    print("\n========== ÉVALUATION GLOBALE ==========")
    print(f"Images évaluées        : {total_images}")
    print(f"Escaliers détectés     : {escaliers_detectes}")
    print(f"Pourcentage de détection : {pourcentage:.2f}% (avec seuil IoU > {seuil_iou})")
    print("========================================")


# =========================
#         MAIN
# =========================

if __name__ == "__main__":
    dossier_images = "Base_Validation"
    dossier_jsons_gt = "json"          # Vérité terrain
    dossier_jsons_pred = "Annotations" # Résultats de prédiction


    # Évaluation globale
    evaluation_globale(dossier_images, dossier_jsons_gt, dossier_jsons_pred, seuil_iou=0.5)
