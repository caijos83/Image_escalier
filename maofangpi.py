#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconnaissance d'Escaliers par Traitement d'Image Basique
Ce script charge une image, la convertit en niveaux de gris, applique un flou gaussien,
effectue le seuillage avec l'algorithme d'Otsu, nettoie l'image par morphologie (ouverture et fermeture),
détecte les contours à l'aide d'un filtre de Sobel et applique une transformée de Hough basique.
Enfin, il affiche les différentes étapes pour visualiser le pipeline.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === 1. Fonctions de convolution et de filtre gaussien ===

def convolve2d(image, kernel):
    """
    Convolution 2D d'une image par un kernel.
    """
    kernel = np.flipud(np.fliplr(kernel))  # Retourner le kernel
    m, n = kernel.shape
    pad_h, pad_w = m // 2, n // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=float)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+m, j:j+n] * kernel)
    return output

def gaussian_kernel(size=5, sigma=1):
    """
    Génère un noyau gaussien de taille donnée et d'un sigma défini.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# === 2. Seuillage d'Otsu ===

def algo_otsu(image):
    """
    Algorithme d'Otsu pour déterminer le seuil optimal.
    L'image doit être en niveaux de gris avec des valeurs [0,255].
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256))
    total = image.size

    current_max, threshold = 0, 0
    sum_total, sumB, weightB = 0, 0, 0
    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weightB += hist[i]
        if weightB == 0:
            continue
        weightF = total - weightB
        if weightF == 0:
            break
        sumB += i * hist[i]
        meanB = sumB / weightB
        meanF = (sum_total - sumB) / weightF
        var_between = weightB * weightF * (meanB - meanF) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i
    return threshold

# === 3. Opérations Morphologiques (Érosion, Dilatation, Ouverture, Fermeture) ===

def dilation(binary, kernel):
    """
    Dilatation d'une image binaire à l'aide d'un noyau booléen.
    """
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=False)
    out = np.zeros_like(binary, dtype=bool)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i:i+kH, j:j+kW]
            if np.any(region[kernel]):
                out[i, j] = True
    return out

def erosion(binary, kernel):
    """
    Érosion d'une image binaire à l'aide d'un noyau booléen.
    """
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=True)
    out = np.zeros_like(binary, dtype=bool)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i:i+kH, j:j+kW]
            if np.all(region[kernel]):
                out[i, j] = True
    return out

def opening(binary, kernel):
    """Ouverture = Érosion puis Dilatation."""
    return dilation(erosion(binary, kernel), kernel)

def closing(binary, kernel):
    """Fermeture = Dilatation puis Érosion."""
    return erosion(dilation(binary, kernel), kernel)

# === 4. Détection de Contours par Sobel ===

def sobel_edges(image):
    """
    Détecte les contours d'une image en appliquant les filtres de Sobel.
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    grad_x = convolve2d(image, Kx)
    grad_y = convolve2d(image, Ky)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_threshold = np.percentile(magnitude, 75)
    edges = magnitude > edge_threshold
    return edges

# === 5. Transformée de Hough (implémentation basique) ===

def hough_transform(edges, theta_res=1, rho_res=1):
    """
    Applique la transformée de Hough sur une image de contours.
    Renvoie l'accumulateur, les vecteurs d'angles et de rho.
    """
    rows, cols = edges.shape
    diag_len = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)) + diag_len)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

# === 6. Pipeline Principal ===

def main():
    # 1. Chargement et Conversion en niveaux de gris
    img = Image.open('images/86.jpg').convert('L')
    gray = np.array(img)

    # 2. Application d'un Flou Gaussien
    kernel_gauss = gaussian_kernel(size=5, sigma=1)
    blurred = convolve2d(gray, kernel_gauss)
    # On convertit en uint8 pour le seuillage
    blurred_uint8 = np.clip(blurred, 0, 255).astype(np.uint8)

    # 3. Seuillage avec l'algorithme d'Otsu
    thresh = algo_otsu(blurred_uint8)
    binary = blurred_uint8 > thresh
    # Inversion éventuelle : ici, on inverse pour que les objets soient à True
    binary = np.invert(binary)

    # 4. Opérations Morphologiques
    kernel_morph = np.ones((3, 3), dtype=bool)
    opened = opening(binary, kernel_morph)
    closed = closing(opened, kernel_morph)

    # 5. Détection de Contours avec Sobel
    edges = sobel_edges(closed.astype(np.float32))

    # 6. Transformée de Hough (détection de lignes)
    accumulator, thetas, rhos = hough_transform(edges)

    # === Visualisation des Étapes ===
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.ravel()
    
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title("Image Grise")
    
    ax[1].imshow(blurred_uint8, cmap='gray')
    ax[1].set_title("Flou Gaussien")
    
    ax[2].imshow(binary, cmap='gray')
    ax[2].set_title("Seuillage (Otsu)")
    
    ax[3].imshow(closed, cmap='gray')
    ax[3].set_title("Morphologie (Ouverture/Fermeture)")
    
    ax[4].imshow(edges, cmap='gray')
    ax[4].set_title("Contours (Sobel)")
    
    ax[5].imshow(accumulator, cmap='hot', aspect='auto')
    ax[5].set_title("Transformée de Hough")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
