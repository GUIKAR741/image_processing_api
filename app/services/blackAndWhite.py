"""."""
import numpy as np
import cv2


def blackAndWhite(imagem):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, blackAndWhiteImage = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage
