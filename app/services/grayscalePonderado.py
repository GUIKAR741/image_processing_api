"""."""


def grayscalePonderado(imagem):
    """."""
    (w, h) = imagem.shape[0:2]
    for x in range(w):
        for y in range(h):
            pxl = imagem[x, y]
            imagem[x, y] = int(
                0.114 * pxl[0] + 0.587 * pxl[1] + 0.299 * pxl[2]
            )
    return imagem
