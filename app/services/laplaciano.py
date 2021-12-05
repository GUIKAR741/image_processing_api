"""."""
from .convolucao import convolucao
import numpy as np
from PIL import ImageFilter


def laplaciano(imagem, bordas):
    """."""
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
    else:
        imagem = convolucao(
            imagem,
            np.array([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1],
            ]))
    return imagem
