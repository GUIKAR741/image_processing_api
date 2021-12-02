"""."""
import cv2
import numpy as np


def escalaLogaritmica(imagem, contrast):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) / 255
    imagem = contrast * np.log2(1 + imagem)
    return imagem * 255
