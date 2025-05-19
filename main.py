import os
import shutil
import cv2
import numpy as np
from reconnaissanceEscalier import (
    traiter_dossier_v1,
    traiter_dossier_v2,
)
from evaluation import (
    load_ground_truth_mask,
    calcul_metrique,
    evaluation_globale
)
"""
Ce fichier python regroupe les fonctions du reconnaissanceEscalier.py, 
et de evaluation.py afin de pouvoir faire une évaluation compléte.
"""


def main():
    # Dossiers et paramètres
    dossier_images = "Base_Test"        # Dossier image à traiter
    dossier_jsons_gt = "json"               # Vérité terrain
    annotations_v1 = "Annotations_v1"      # Résultats du traitement v1
    annotations_v2 = "Annotations_v2"      # Résultats du traitement v2
    annotations_final = "Annotations_final" # Fusion des deux
    seuil_iou = 0.5

    # Étape 1: traitement initial (version Gaussian)
    print("===== Traitement v1 =====")
    traiter_dossier_v1(dossier_images, annotations_v1)

    # Étape 2: détection des images échouées (IoU < seuil)
    print("\n===== Sélection des échecs =====")
    valid_ext = ('.jpg', '.jpeg', '.png')
    images = sorted([f for f in os.listdir(dossier_images) if f.endswith(valid_ext)])
    images_echouees = []

    for img in images:
        nom = os.path.splitext(img)[0]
        gt_json = os.path.join(dossier_jsons_gt, f"{nom}.json")
        pred_json = os.path.join(annotations_v1, f"{nom}.json")

        if not os.path.exists(gt_json) or not os.path.exists(pred_json):
            images_echouees.append(img)
            continue

        image_path = os.path.join(dossier_images, img)
        image = cv2.imread(image_path)
        gt_mask = load_ground_truth_mask(gt_json, image.shape)
        pred_mask = load_ground_truth_mask(pred_json, image.shape)
        scores = calcul_metrique(gt_mask, pred_mask)

        if scores['IoU'] < seuil_iou:
            images_echouees.append(img)

    print(f"Images échouées (IoU < {seuil_iou}): {len(images_echouees)}")

    # Étape 3: retraitement v2 des images échouées (version Egalisation)
    if images_echouees:
        temp_folder = "images_echouees"
        os.makedirs(temp_folder, exist_ok=True)
        for img in images_echouees:
            shutil.copy(
                os.path.join(dossier_images, img),
                os.path.join(temp_folder, img)
            )
        print("\n===== Traitement v2 sur échecs =====")
        traiter_dossier_v2(temp_folder, annotations_v2)
    else:
        print("Aucun échec détecté, pas de retraitement v2.")

    # Étape 4: fusion des annotations
    print("\n===== Fusion des annotations =====")
    os.makedirs(annotations_final, exist_ok=True)
    # Copier tout v1
    for filename in os.listdir(annotations_v1):
        shutil.copy(
            os.path.join(annotations_v1, filename),
            os.path.join(annotations_final, filename)
        )
    # Remplacer par les annotations v2 pour les images échouées
    for img in images_echouees:
        nom = os.path.splitext(img)[0]
        src = os.path.join(annotations_v2, f"{nom}.json")
        dst = os.path.join(annotations_final, f"{nom}.json")
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Étape 5: évaluation globale finale
    print("\n===== Évaluation globale finale =====")
    evaluation_globale(dossier_images, dossier_jsons_gt, annotations_final, seuil_iou)



    # Étape 6: statistiques IoU
    print("\n===== Statistiques IoU =====")
    ious_total = []
    ious_succes = []
    ious_echecs = []
    for img in images:
        nom = os.path.splitext(img)[0]
        gt_json = os.path.join(dossier_jsons_gt, f"{nom}.json")
        pred_json = os.path.join(annotations_final, f"{nom}.json")
        if not os.path.exists(gt_json) or not os.path.exists(pred_json):
            continue
        image_path = os.path.join(dossier_images, img)
        image = cv2.imread(image_path)
        gt_mask = load_ground_truth_mask(gt_json, image.shape)
        pred_mask = load_ground_truth_mask(pred_json, image.shape)
        scores = calcul_metrique(gt_mask, pred_mask)
        iou = scores['IoU']
        ious_total.append(iou)
        if iou >= seuil_iou: 
            ious_succes.append(iou)
        else:
            ious_echecs.append(iou)
    if ious_total:
        print(f"IoU moyen total   : {np.mean(ious_total):.4f} (n={len(ious_total)})")
        print(f"IoU moyen succès  : {np.mean(ious_succes):.4f} (n={len(ious_succes)})")
        print(f"IoU moyen échecs  : {np.mean(ious_echecs):.4f} (n={len(ious_echecs)})")
    else:
        print("Aucun IoU calculé.")

    # Étape 7: moyenne F1 et nombre d'images avec F1 >= seuil_iou
    print("\n===== Statistiques F1 =====")
    f1s = []
    for img in images:
        nom = os.path.splitext(img)[0]
        gt_json = os.path.join(dossier_jsons_gt, f"{nom}.json")
        pred_json = os.path.join(annotations_final, f"{nom}.json")
        if not os.path.exists(gt_json) or not os.path.exists(pred_json):
            continue
        image_path = os.path.join(dossier_images, img)
        image = cv2.imread(image_path)
        gt_mask = load_ground_truth_mask(gt_json, image.shape)
        pred_mask = load_ground_truth_mask(pred_json, image.shape)
        scores = calcul_metrique(gt_mask, pred_mask)
        f1s.append(scores['F1-score'])
    if f1s:
        mean_f1 = np.mean(f1s)
        count_high_f1 = sum(1 for val in f1s if val >= seuil_iou)
        print(f"F1 moyen           : {mean_f1:.4f} (n={len(f1s)})")
        print(f"Images F1 >= {seuil_iou} : {count_high_f1}")
    else:
        print("Aucun F1 calculé.")

if __name__ == "__main__":
    main()
