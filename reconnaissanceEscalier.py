import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import json
import math
import os


def image_gray(image_path):
    """
    Convertit une image couleur en niveaux de gris en utilisant la formule de luminance.
    """
    img = mplimg.imread(image_path)

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    gray_image = np.dot(img[..., :3], [0.299, 0.587, 0.114])

    return gray_image.astype(np.uint8)


def algo_otsu(image_path):
    """
    Applique l'algorithme de seuillage d'Otsu sur une image en niveaux de gris.
    """
    img = image_gray(image_path)

    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    total_pixels = img.size
    prob = hist / total_pixels

    cumulative_sum = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * np.arange(256))
    global_mean = cumulative_mean[-1]

    max_variance = 0
    optimal_threshold = 0

    for t in range(1, 256):
        weight_bg = cumulative_sum[t]
        weight_fg = 1 - weight_bg
        mean_bg = cumulative_mean[t] / weight_bg if weight_bg != 0 else 0
        mean_fg = (cumulative_mean[-1] - cumulative_mean[t]) / weight_fg if weight_fg != 0 else 0
        variance_inter_class = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance_inter_class > max_variance:
            max_variance = variance_inter_class
            optimal_threshold = t

    binary_image = img > optimal_threshold
    binary_image = binary_image.astype(np.uint8) * 255  # Pour affichage OpenCV

    return binary_image, img  # On retourne aussi l'image en niveaux de gris





def draw_hough_lines(img, lines, color=(0, 0, 255), thickness=2):
    """
        Dessine les segments de lignes détectés par HoughLinesP sur l'image.

        Paramètres :
            img (ndarray) : Image d'origine, en niveaux de gris ou en couleur.
            lignes (ndarray) : Résultat retourné par HoughLinesP.
            couleur (tuple) : Couleur des lignes tracées (par défaut : rouge en BGR).
            epaisseur (int) : Épaisseur des lignes (par défaut : 2).

        Retour :
            img_avec_lignes (ndarray) : Copie de l'image avec les lignes tracées.
    """
   
    img_with_lines = img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  
            cv2.line(img_with_lines, (x1, y1), (x2, y2), color, thickness)

    return img_with_lines

    
def getDegree(x1, y1, x2, y2):
    dy = y2 - y1
    dx = x2 - x1

    angle_rad = math.atan(dy/dx)

    angle_deg = math.degrees(angle_rad)
    

    return angle_deg

def get_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def merge_similar_lines(lines, angle_threshold=5, distance_threshold=20):
    merged = []
    mergedLines=[]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = get_angle(x1, y1, x2, y2)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        matched = False
        for group in merged:
            g_angle, g_cx, g_cy, group_lines = group

            if abs(angle - g_angle) < angle_threshold and \
               math.hypot(center_x - g_cx, center_y - g_cy) < distance_threshold:
                group_lines.append((x1, y1, x2, y2))

                group[0] = (g_angle + angle) / 2
                group[1] = (g_cx + center_x) / 2
                group[2] = (g_cy + center_y) / 2
                matched = True
                break

        if not matched:
            merged.append([angle, center_x, center_y, [(x1, y1, x2, y2)]])
            mergedLines.append(line)

    return mergedLines

def sort_lines_by_y1(lines):
    return sorted(lines, key=lambda line: line[0][1])

def getLinesP(img,width):
    houghLines = cv2.HoughLinesP(img, 
                        rho=1,                      # la précision en distance, en pixels
                        theta=np.pi/180,    # la précision angulaire, détecter une fois tous les °
                        threshold=75,         # au moins les points existent
                        minLineLength=width/6,   
                        maxLineGap=20)
    lines=merge_similar_lines(houghLines)
    total=0
    nb=0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        degree=getDegree(x1,y1,x2,y2)
        if abs(degree)<=75:
            total=total+degree
            nb=nb+1
    if nb == 0:
        return []
    ave = total / nb


    linesP=[]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        degree=getDegree(x1,y1,x2,y2)
        if abs(degree-ave)<=25:
            linesP.append(line)

    linesP=sort_lines_by_y1(linesP)

    return linesP


def get_corners_from_lines(lines):
    """
    通过线段端点直接定位四个角点
    :param lines: 线段列表，格式为 [np.array([[x1,y1,x2,y2]]), ...]
    :return: 四个角点坐标 (top_left, top_right, bottom_left, bottom_right)
    """
    # 将所有端点展平为点列表
    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_points.extend([(x1, y1), (x2, y2)])


    # 去重处理
    unique_points = list(set(all_points))

    # 定义角落评分规则
    def score_lt(p): return p[0] + p[1]     # 左上角评分 (x+y 最小)
    def score_rt(p): return p[0] - p[1]     # 右上角评分 (x-y 最大)
    def score_lb(p): return p[0] - p[1]     # 左下角评分 (x-y 最小)
    def score_rb(p): return p[0] + p[1]     # 右下角评分 (x+y 最大)

    # 找到最符合每个角落特征的点
    top_left     = min(unique_points, key=lambda p: score_lt(p))
    top_right    = max(unique_points, key=lambda p: score_rt(p))
    bottom_left  = min(unique_points, key=lambda p: score_lb(p))
    bottom_right = max(unique_points, key=lambda p: score_rb(p))

    return top_left, top_right, bottom_right, bottom_left


    
def get_rectangle_from_lines(lines):
    """
    Génére un quadrilatère (presque rectangle) à partir des lignes détectées sur les escaliers.
    """
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))

    # Trier les points par Y (du plus haut au plus bas)
    points = sorted(points, key=lambda p: p[1])

    # Prendre les points les plus hauts (haut gauche et droit)
    top_points = sorted(points[:10], key=lambda p: p[0])  # 10 plus hauts
    top_left = top_points[0]
    top_right = top_points[-1]

    # Prendre les points les plus bas (bas gauche et droit)
    bottom_points = sorted(points[-10:], key=lambda p: p[0])  # 10 plus bas
    bottom_left = bottom_points[0]
    bottom_right = bottom_points[-1]

    return [top_left, top_right, bottom_right, bottom_left]
    
    
# =========================
#         TRAITER IMAGE VERSION GAUSSIAN
# =========================


def traiter_dossier_v1(dossier_images, dossier_annotations):
    os.makedirs(dossier_annotations, exist_ok=True)
    fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith('.jpg')])

    for fichier in fichiers:
        image_path = os.path.join(dossier_images, fichier)
        nom_base = os.path.splitext(fichier)[0]
        json_path = os.path.join(dossier_annotations, f"{nom_base}.json")

        print(f"Traitement de {fichier}...")

        binary_image, gray_img = algo_otsu(image_path)

        gaussien = cv2.GaussianBlur(gray_img, (7, 7), 0)
        contour = cv2.Canny(gaussien, 100, 180)
        noyau = np.ones((3, 3), np.uint8)
        dilated_contour = cv2.dilate(contour, noyau, iterations=3)

        width = contour.shape[1]
        lines = getLinesP(dilated_contour, width)

        if not lines:
            print(f"[!] Aucune ligne détectée pour {fichier}.")
            continue

        points = get_corners_from_lines(lines)

        #points = get_rectangle_from_lines(lines)
        points_formatted = [[int(p[0]), int(p[1])] for p in points]

        annotation = {
            "shapes": [
                {
                    "label": "escalier",
                    "points": points_formatted,
                    "shape_type": "polygon"
                }
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=4)

        print(f"Annotation sauvegardée : {json_path}")
        
        
# =========================
#         TRAITER IMAGE VERSION EGALISED
# =========================


def traiter_dossier_v2(dossier_images, dossier_annotations):
    os.makedirs(dossier_annotations, exist_ok=True)
    fichiers = sorted([f for f in os.listdir(dossier_images) if f.endswith('.jpg')])

    for fichier in fichiers:
        image_path = os.path.join(dossier_images, fichier)
        nom_base = os.path.splitext(fichier)[0]
        json_path = os.path.join(dossier_annotations, f"{nom_base}.json")

        print(f"Traitement de {fichier}...")

        binary_image, gray_img = algo_otsu(image_path)
        #égaliser l'image
        equalized = cv2.equalizeHist(gray_img)
        
    
        #gaussien = cv2.GaussianBlur(gray_img, (7, 7), 0)
        contour = cv2.Canny(equalized, 100, 180)

        noyau = np.ones((3, 3), np.uint8)
        
        dilated_contour = cv2.dilate(contour,noyau, iterations=4)
        closed_contour = cv2.erode(dilated_contour, noyau, iterations=3)

        width = contour.shape[1]
        lines = getLinesP(closed_contour, width)

    
        if not lines:
            print(f"[!] Aucune ligne détectée pour {fichier}.")
            continue
            
            

        points = get_corners_from_lines(lines)

        #points = get_rectangle_from_lines(lines)
        points_formatted = [[int(p[0]), int(p[1])] for p in points]

        annotation = {
            "shapes": [
                {
                    "label": "escalier",
                    "points": points_formatted,
                    "shape_type": "polygon"
                }
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=4)

        print(f"Annotation sauvegardée : {json_path}")



# === Lancer le traitement ===
if __name__ == "__main__":
      
    dossier_images = "Base_Validation"
    dossier_annotations = "Annotations"

    traiter_dossier_v1(dossier_images, dossier_annotations)
    traiter_dossier_v2(dossier_images, dossier_annotations)


