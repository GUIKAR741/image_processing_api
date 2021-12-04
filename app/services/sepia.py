"""."""
import numpy as np


def sepia(imagem):
    """."""
    (w, h) = imagem.shape[0:2]
    for x in range(w):
        for y in range(h):
            b, g, r = imagem[x, y]
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            tr = 255 if tr > 255 else tr
            tg = 255 if tg > 255 else tg
            tb = 255 if tb > 255 else tb
            imagem[x, y] = (tb, tg, tr)
    return imagem
