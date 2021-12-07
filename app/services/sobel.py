"""."""
from .convolucao import convolucao
import numpy as np
import cv2


def normalizar(img):
    """."""
    img = img - img.min()
    return img / img.max() * 255


def sobel(imagem, bordas, normaliza):
    """."""
    imagem_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if bordas == 0:
        imagem = convolucao(
            imagem,
            np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]),
            normaliza
        )
    elif bordas == 1:
        imagem = convolucao(
            imagem,
            np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ]),
            normaliza
        )
    elif bordas == 2:
        imagem = np.abs(convolucao(
            imagem,
            np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]),
            normaliza)) + \
            np.abs(convolucao(
                imagem,
                np.array([
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1],
                ]),
                normaliza
            )) if normaliza == 0 else convolucao(
                imagem,
                np.array([
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1],
                ]),
                normaliza) + convolucao(
                    imagem,
                    np.array([
                        [-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1],
                    ]),
                    normaliza)
    elif bordas == 5:
        imagem = imagem_original + convolucao(
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
    return normalizar(imagem) if normaliza > 0 else imagem
