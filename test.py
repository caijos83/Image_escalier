import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import json


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

    
    
def getSlope(x1,y1,x2,y2):
    if x2 == x1:
               return float('inf')
    return (y2-y1)/(x2-x1)

def getLinesP(img,width):
    lines = cv2.HoughLinesP(img, 
                        rho=1,                      # la précision en distance, en pixels
                        theta=np.pi/180,    # la précision angulaire, détecter une fois tous les °
                        threshold=75,         # au moins les points existent
                        minLineLength=width/4,   
                        maxLineGap=20)

    linesXP=[]    #k>0
    linesXM=[]    #k<0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope=getSlope(x1,y1,x2,y2)

        if slope<=1 and slope>=0:
            linesXP.append(line)

        if slope<=0 and slope>=-1:
            linesXM.append(line)

    if len(linesXM)==0 and len(linesXM)==0 :
        return None

    linesP=[]

    if len(linesXM)>len(linesXP) :
        total=0
        for line in linesXM:
            x1, y1, x2, y2 = line[0]
            slope=getSlope(x1,y1,x2,y2)
            total=total+slope

        ave=total/len(linesXM)

        for line in linesXM:
            x1, y1, x2, y2 = line[0]
            slope=getSlope(x1,y1,x2,y2)
            if abs(slope-ave)<=0.8:
                linesP.append(line)
        return linesP

    if len(linesXP)>=len(linesXM) :
        total=0
        for line in linesXP:
            x1, y1, x2, y2 = line[0]
            slope=getSlope(x1,y1,x2,y2)
            total=total+slope

        ave=total/len(linesXP)

        for line in linesXP:
            x1, y1, x2, y2 = line[0]
            slope=getSlope(x1,y1,x2,y2)
            if abs(slope-ave)<=0.8:
                linesP.append(line)
        return linesP

# === Main ===
image_path = "images/25.jpg"

# 1. Otsu + récupération image niveaux de gris
binary_image, gray_img = algo_otsu(image_path)

# 2. Appliquer flou gaussien et Canny
gaussien = cv2.GaussianBlur(gray_img, (7, 7), 0)
contour = cv2.Canny(gaussien, 100, 180)

noyau= np.ones((3, 3), np.uint8)
dilated_contour = cv2.dilate(contour,noyau, iterations=3)

# 3. Redimensionner et afficher avec OpenCV
resized_contour = cv2.resize(dilated_contour, (600, 600))
cv2.imshow("Contours dilated", resized_contour)


# 4.
width = contour.shape[1]

lines = getLinesP(dilated_contour, width)
"""
lines = cv2.HoughLinesP(dilated_contour, 
                        rho=1,                      # la précision en distance, en pixels
                        theta=np.pi/180,    # la précision angulaire, détecter une fois tous les °
                        threshold=75,         # au moins les points existent
                        minLineLength=width/4,   
                        maxLineGap=20)
             
             
"""
original_img = cv2.imread(image_path)
imgLine = draw_hough_lines(original_img, lines)


cv2.imshow("Lines", cv2.resize(imgLine, (600, 600)))



# 5. Afficher image binaire Otsu avec matplotlib

plt.imshow(binary_image, cmap='gray')
plt.title("Image binarisée - Otsu")
plt.axis('off')
plt.show()


# 5. Sauvegardes
plt.imsave("binary_image.jpg", binary_image, cmap='gray')
#cv2.imwrite("contours_canny_.jpg", contour)

cv2.waitKey(0)
cv2.destroyAllWindows()
