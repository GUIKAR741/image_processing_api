"""."""
import numpy as np
import cv2
from PIL import Image
import random


def bitplanes(im):
    """."""
    data = np.array(im)
    out = []
    # cria uma imagem para cada k bit
    for k in range(7, -1, -1):
        # extrai o k-esimo bit (de 0 a 7)
        res = data // 2**k & 1
        out.append(res * 255)
    # empilha as imagens geradas
    b = np.hstack(out)
    return b


def bitplanesDecrypt(im):
    """."""
    data = np.array(im)
    out = []
    for k in range(7, -1, -1):
        # extrai o k-esimo bit (de 0 a 7)
        res = data // 2**k & 1
        out = res * 255
    return out


def estenografiaLSB(imagem1, imagem2):
    """."""
    imagem1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
    imagem2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)
    imagem2 = Image.fromarray(imagem2)
    imagem2 = imagem2.resize(imagem1.shape[::-1])
    imagem2 = np.array(imagem2, dtype=np.uint8)
    _, imagem2 = cv2.threshold(imagem2, 127, 1, cv2.THRESH_BINARY)
    resultado = imagem1 & ~1 | imagem2
    cv2.imwrite("encript.png", resultado)
    return resultado


def estenografiaLSBDecrypt(imagem1):
    """."""
    imagem1 = bitplanesDecrypt(imagem1)
    cv2.imwrite("decript.png", imagem1)
    return imagem1


def estenografia(imagem1, imagem2):
    """."""
    imagem2 = Image.fromarray(imagem2)
    imagem2 = imagem2.resize(imagem1.shape[:2][::-1])
    imagem2 = np.array(imagem2, dtype=np.uint8)
    for i in range(imagem1.shape[0]):
        for j in range(imagem1.shape[1]):
            for l in range(len(imagem1[i][j])):
                v1 = format(imagem1[i][j][l], '08b')
                v2 = format(imagem2[i][j][l], '08b')
                v3 = v1[:4] + v2[:4]
                imagem1[i][j][l] = int(v3, 2)
    cv2.imwrite("encript.png", imagem1)
    return imagem1


def estenografiaDecrypt(imagem1):
    """."""
    imagem2 = np.zeros(imagem1.shape, np.uint8)
    for i in range(imagem1.shape[0]):
        for j in range(imagem1.shape[1]):
            for l in range(len(imagem1[i][j])):
                v1 = format(imagem1[i][j][l], '08b')
                v3 = v1[4:] + chr(random.randint(0, 1) + 48) * 4
                imagem2[i][j][l] = int(v3, 2)
    cv2.imwrite("decript.png", imagem2)
    return imagem2
