"""."""
import math
import cv2
import numpy as np


def gammaCorrection(imagem, gamma, contrast):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) / 255
    imagem = contrast * (imagem ** gamma)
    return imagem * 255
