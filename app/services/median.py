"""."""
from .convolucao import convolucao
import numpy as np


def median(imagem, tamanho):
    """."""
    imagem = convolucao(
        imagem,
        np.ones((tamanho, tamanho)) / (tamanho * tamanho)
    )
    return imagem
