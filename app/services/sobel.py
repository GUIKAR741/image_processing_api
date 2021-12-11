"""."""
from .convolucao import convolucao, normalizaImagem
import numpy as np


def normalizar(img):
    """."""
    img = img - img.min()
    return img / img.max() * 255


def sobel(imagem, bordas, normaliza):
    """."""
    normaliza = 1 if normaliza == 0 else 0
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
        imagem = convolucao(
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
        imagem = normalizaImagem(imagem) * 255 if normaliza == 0 else imagem
    return imagem if normaliza == 0 else imagem * 255
