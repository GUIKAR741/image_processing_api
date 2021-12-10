"""."""
from .convolucao import convolucao
import numpy as np


def mean(imagem, tamanho):
    """."""
    imagem = convolucao(
        imagem,
        np.ones((tamanho, tamanho)),
        func=lambda m, k: np.sum(m) / (tamanho ** 2)
    )
    return imagem


def geometric(imagem, tamanho):
    """."""
    imagem = convolucao(
        imagem,
        np.ones((tamanho, tamanho)),
        func=lambda m, k: np.prod(m) ** (1 / (tamanho ** 2))
    )
    return imagem


def harmonic(imagem, tamanho):
    """."""
    def harmonicCalc(m, k):
        m = m[m != 0]
        return (tamanho ** 2) / np.sum(1 / m)
    imagem = convolucao(
        imagem,
        np.ones((tamanho, tamanho)),
        func=harmonicCalc
    )
    return imagem


def contraHarmonic(imagem, tamanho, q):
    """."""
    def contraHarmonicCalc(m, k):
        m = m[m != 0]
        return np.sum(np.power(m, q + 1)) / np.sum(np.power(m, q))
    imagem = convolucao(
        imagem,
        np.ones((tamanho, tamanho)),
        func=contraHarmonicCalc
    )
    return imagem
