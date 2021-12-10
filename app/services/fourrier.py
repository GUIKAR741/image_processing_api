"""."""
import cv2
import numpy as np
import math


def gaussian_kernel(sigma, size):
    """."""
    mu = np.floor([size / 2, size / 2])
    size = int(size)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(
                -(0.5 / (sigma * sigma)) * (
                    np.square(i - mu[0]) + np.square(j - mu[0])
                )) / np.sqrt(2 * math.pi * sigma * sigma)
    kernel = kernel / np.sum(kernel)
    return kernel


def fourrierFiltros(
        imagem,
        mostraTransformada,
        clip,
        tipo,
        raio,
        sigma,
        raioInterno
):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagemDobrada = np.zeros((imagem.shape[0] * 2, imagem.shape[1] * 2))
    imagemDobrada[0:imagem.shape[0], 0:imagem.shape[1]] = imagem
    imagem, imagemDobrada = imagemDobrada, imagem
    imax = imagem / imagem.max()
    it = np.fft.fft2(imax)
    it = np.fft.fftshift(it)
    if mostraTransformada > 0:
        it = np.abs(it)
        it = it.clip(0, clip)
        it = it / clip
    if tipo == None:
        ...
    elif tipo == 1 or tipo == 2:
        if raio != None and raio > 0:
            y, x = np.ogrid[-raio:raio + 1, -raio:raio + 1]
            disk = x**2 + y**2 <= raio**2
            disk = disk.astype(float)

            disco = np.zeros(imagem.shape)
            disco[imagem.shape[0] // 2 - raio:imagem.shape[0] // 2 + raio + 1,
                  imagem.shape[1] // 2 - raio:imagem.shape[1] // 2 + raio + 1] = disk

            it = it * ((1 - disco) if tipo == 1 else disco)
    elif tipo == 3 or tipo == 4:
        if raio != None and sigma != None and raio > 0:
            kernel = gaussian_kernel(sigma, raio * 2)
            kernel = kernel / kernel.max()

            kernelAplicado = np.zeros(imagem.shape)
            kernelAplicado[imagem.shape[0] // 2 - raio:imagem.shape[0] // 2 + raio,
                           imagem.shape[1] // 2 - raio:imagem.shape[1] // 2 + raio] = kernel

            it = it * ((1 - kernelAplicado) if tipo == 3 else kernelAplicado)
    elif tipo == 5 or tipo == 6:
        if raio != None and raioInterno != None and raio > 0 and \
                raioInterno > 0 and raioInterno < raio:
            y, x = np.ogrid[-raio:raio + 1, -raio:raio + 1]
            disk = x**2 + y**2 <= raio**2
            disk = disk.astype(float)

            y, x = np.ogrid[-raioInterno:raioInterno + 1,
                            -raioInterno:raioInterno + 1]
            disk2 = x**2 + y**2 <= raioInterno**2
            disk2 = 1 - disk2.astype(float)

            disco = np.zeros(imagem.shape)
            disco[imagem.shape[0] // 2 - raio:imagem.shape[0] // 2 + raio + 1,
                  imagem.shape[1] // 2 - raio:imagem.shape[1] // 2 + raio + 1] = disk

            disco2 = np.ones(imagem.shape)
            disco2[imagem.shape[0] // 2 - raioInterno:
                   imagem.shape[0] // 2 + raioInterno + 1,
                   imagem.shape[1] // 2 - raioInterno:
                   imagem.shape[1] // 2 + raioInterno + 1] = disk2

            disco = disco * disco2
            it = it * ((1 - disco) if tipo == 5 else disco)
    elif tipo == 7 or tipo == 8:
        if raio != None and sigma != None and raioInterno != None and \
                raio > 0 and raioInterno > 0 and raioInterno < raio:
            kernel = gaussian_kernel(sigma, raio * 2)
            kernel = kernel / kernel.max()
            y, x = np.ogrid[-raioInterno:raioInterno + 1,
                            -raioInterno:raioInterno + 1]
            disk = x**2 + y**2 <= raioInterno**2
            disk = 1 - disk.astype(float)

            kernelAplicado = np.zeros(imagem.shape)
            kernelAplicado[imagem.shape[0] // 2 - raio:imagem.shape[0] // 2 + raio,
                           imagem.shape[1] // 2 - raio:imagem.shape[1] // 2 + raio] = kernel
            kernelAplicado1 = np.ones(imagem.shape)
            kernelAplicado1[imagem.shape[0] // 2 - raioInterno:
                            imagem.shape[0] // 2 + raioInterno + 1,
                            imagem.shape[1] // 2 - raioInterno:
                            imagem.shape[1] // 2 + raioInterno + 1] = disk
            kernelAplicado = kernelAplicado * kernelAplicado1
            it = it * ((1 - kernelAplicado) if tipo == 7 else kernelAplicado)
    if mostraTransformada == 0:
        it = np.fft.ifft2(it)
        it = np.abs(it)
        it = it / it.max()
        it = it[0:imagemDobrada.shape[0], 0:imagemDobrada.shape[1]]
    return it * 255


def fourrierManual(imagem, mostraTransformada, espaco, clip):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagemDobrada = np.zeros((imagem.shape[0] * 2, imagem.shape[1] * 2))
    imagemDobrada[0:imagem.shape[0], 0:imagem.shape[1]] = imagem
    imagem, imagemDobrada = imagemDobrada, imagem
    imax = imagem / imagem.max()
    it = np.fft.fft2(imax)
    it = np.fft.fftshift(it)
    if mostraTransformada > 0:
        it = np.abs(it)
        it = it.clip(0, clip)
        it = it / clip
    if espaco != None:
        f = np.ones(it.shape)
        for i in espaco:
            f[i[0] * 2:i[1] * 2, i[2] * 2:i[3] * 2] = 0
        it = it * f
    if mostraTransformada == 0:
        it = np.fft.ifft2(it)
        it = np.abs(it)
        it = it / it.max()
        it = it[0:imagemDobrada.shape[0], 0:imagemDobrada.shape[1]]
    return it * 255


def fourrier(imagem):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    if imagem.shape[0] > 50 or imagem.shape[1] > 50:
        imagem = cv2.resize(imagem, (50, 50))
    h = imagem.shape[0]
    w = imagem.shape[1]
    imagem_saida = np.zeros((h, w), complex)
    for m in range(0, h):
        for n in range(0, w):
            for x in range(0, h):
                for y in range(0, w):
                    imagem_saida[m][n] += imagem[x][y] * np.exp(-1j * 2 * math.pi * (m * x / h + n * y / w))
    imagem_saida = np.fft.fftshift(imagem_saida)
    imagem_saida = np.abs(imagem_saida)
    imagem_saida = imagem_saida - imagem_saida.min()
    imagem_saida = imagem_saida / imagem_saida.max()
    return imagem_saida * 255
