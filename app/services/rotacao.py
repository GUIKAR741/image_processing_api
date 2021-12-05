"""."""
import numpy as np
import math


def rotacao(imagem, angulo):
    """."""
    angulo = math.radians(angulo)
    cosine = math.cos(angulo)
    sine = math.sin(angulo)
    h = imagem.shape[0]
    w = imagem.shape[1]

    nova_h = round(abs(imagem.shape[0] * cosine) + abs(imagem.shape[1] * sine)) + 1
    nova_w = round(abs(imagem.shape[1] * cosine) + abs(imagem.shape[0] * sine)) + 1
    imagem_saida = np.zeros((nova_h, nova_w, imagem.shape[2]))

    centro_original_h = round(((imagem.shape[0] + 1) / 2) - 1)
    centro_original_w = round(((imagem.shape[1] + 1) / 2) - 1)

    novo_centro_h = round(((nova_h + 1) / 2) - 1)
    novo_centro_w = round(((nova_w + 1) / 2) - 1)

    for i in range(h):
        for j in range(w):
            y = imagem.shape[0] - 1 - i - centro_original_h
            x = imagem.shape[1] - 1 - j - centro_original_w

            novo_y = round(-x * sine + y * cosine)
            novo_x = round(x * cosine + y * sine)

            novo_y = novo_centro_h - novo_y
            novo_x = novo_centro_w - novo_x

            if 0 <= novo_x < nova_w and 0 <= novo_y < nova_h and novo_x >= 0 and novo_y >= 0:
                imagem_saida[novo_y, novo_x, :] = imagem[i, j, :]
    return imagem_saida
