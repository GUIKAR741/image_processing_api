"""."""
from math import sqrt, pow


def chromakey(imagem, r, g, b, distancia):
    """."""
    (w, h) = imagem.shape[0:2]
    for x in range(w):
        for y in range(h):
            bi, gi, ri = imagem[x, y]
            d = sqrt(pow(ri - r, 2) + pow(gi - g, 2) + pow(bi - b, 2))
            imagem[x, y] = (255, 255, 255) if d < distancia else (bi, gi, ri)
    return imagem
