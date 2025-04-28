import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import json
import math


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
    ave=total/nb
    ave=total/nb


    linesP=[]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        degree=getDegree(x1,y1,x2,y2)
        if abs(degree-ave)<=20:
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

    return top_left, top_right, bottom_left, bottom_right


# === Main ===
image_path = "Base_Validation\\16.jpg"

# 1. Otsu + récupération image niveaux de gris
binary_image, gray_img = algo_otsu(image_path)

# 2. Appliquer flou gaussien et Canny
gaussien = cv2.GaussianBlur(gray_img, (7, 7), 0)
contour = cv2.Canny(gaussien, 100, 180)

noyau= np.ones((3, 3), np.uint8)
dilated_contour = cv2.dilate(gaussien,noyau, iterations=4)
closed_contour = cv2.erode(dilated_contour, noyau, iterations=3)

# 3. Redimensionner et afficher avec OpenCV
resized_contour = cv2.resize(closed_contour, (600, 600))
cv2.imshow("Contours closed",resized_contour)


# 4.
width = contour.shape[1]

lines = getLinesP(closed_contour, width)

"""
lines = cv2.HoughLinesP(dilated_contour, 
                        rho=1,                      # la précision en distance, en pixels
                        theta=np.pi/180,    # la précision angulaire, détecter une fois tous les °
                        threshold=75,         # au moins les points existent
                        minLineLength=width/6,   
                        maxLineGap=20)
"""             
         
original_img = cv2.imread(image_path)
imgLine = draw_hough_lines(original_img, lines)




print("Lines found:", lines)
if lines is not None:
    top_left, top_right, bottom_left, bottom_right = get_corners_from_lines(lines)

    print("Top Left:", top_left)
    print("Top Right:", top_right)
    print("Bottom Left:", bottom_left)
    print("Bottom Right:", bottom_right)
    for point in [top_left, top_right, bottom_left, bottom_right]:
        cv2.circle(imgLine, point, radius=20, color=(255, 0, 0), thickness=-1)
  
else:
    print("No lines detected.")


cv2.imshow("Lines", cv2.resize(imgLine, (600, 600)))



# 5. Afficher image binaire Otsu avec matplotlib

plt.imshow(binary_image, cmap='gray')
plt.title("Image binarisée - Otsu")
plt.axis('off')
plt.show()


# 5. Sauvegardes
#plt.imsave("binary_image.jpg", binary_image, cmap='gray')
#cv2.imwrite("contours_canny_.jpg", contour)

cv2.waitKey(0)
cv2.destroyAllWindows()

