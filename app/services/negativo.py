"""."""
import cv2


def negativo(imagem, isRGB):
    """."""
    if isRGB == 0:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem = 1 - (imagem / 255)
    imagem *= 255
    return imagem
