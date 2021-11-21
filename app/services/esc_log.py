"""."""
import cv2
import numpy as np


def escalaLogaritmica(imagem, contrast):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) / 255
    imagem = contrast * np.log2(1 + imagem)
    # c = 255 / np.log(1 + np.max(imagem))
    # imagem = c * (np.log(imagem + 1))
    # imagem = np.array(imagem, dtype=np.uint8)
    return imagem * 255
