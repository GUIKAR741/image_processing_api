"""."""
import numpy as np

def grayscaleMedia(imagem):
    """."""
    (w, h) = imagem.shape[0:2]
    for x in range(w):
        for y in range(h):
            pxl = imagem[x, y]
            imagem[x, y] = sum(pxl) // 3
    return imagem
