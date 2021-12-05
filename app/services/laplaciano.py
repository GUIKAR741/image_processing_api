"""."""
from .convolucao import convolucao
import numpy as np
import cv2


def laplaciano(imagem, bordas):
    """."""
    imagem_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if bordas == 0:
        imagem = convolucao(
            imagem,
            np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]))
    elif bordas == 1:
        imagem = convolucao(
            imagem,
            np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ]))
    elif bordas == 3:
        imagem = convolucao(
            imagem,
            np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ]))
    elif bordas == 4:
        imagem = convolucao(
            imagem,
            np.array([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1],
            ]))
    else:
        highBoost = imagem_original - convolucao(
            imagem,
            np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]) / 16)
        imagem = imagem_original + 1.5 * highBoost
    return imagem
