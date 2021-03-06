"""."""
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def inicia_histograma():
    """."""
    hist_dct = {str(i): 0
                for i in range(0, 256)}
    return hist_dct


def conta_intensidade_pixels(hist, img):
    """."""
    for row in img:
        for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
    return hist


def plot_hist(hist, espaco):
    """."""
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(1, 1, 1)
    histValues = list(map(lambda a: a / 255, hist.values()))
    if espaco != None:
        for i in espaco:
            for j in range(i[0], i[1] + 1):
                histValues[j] = i[2]
    axis.bar(hist.keys(), histValues, color="red")
    lineY = list(range(0, 256))
    if espaco != None:
        for i in espaco:
            for j in range(i[0], i[1] + 1):
                lineY[j] = i[2]
    axis.plot(list(range(0, 256)), lineY, color="black")
    axis.set_xlabel("Níveis intensidade")
    axis.set_xlim([0, 255])
    axis.set_ylim([0, 255])
    axis.axes.xaxis.set_ticks([])
    axis.axes.yaxis.set_ticks(list(range(0, 256, 15)))
    axis.grid(True)
    return figure


def plot_hist_rgb(r, g, b):
    """."""
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(r.keys(), list(map(lambda a: a / 255, r.values())), color="blue")
    axis.bar(g.keys(), list(map(lambda a: a / 255, g.values())), color="green")
    axis.bar(b.keys(), list(map(lambda a: a / 255, b.values())), color="red")
    axis.plot(list(range(0, 256)), list(range(0, 256)), color="black")
    axis.set_xlabel("Níveis intensidade")
    axis.set_xlim([0, 255])
    axis.set_ylim([0, 255])
    axis.axes.xaxis.set_ticks([])
    axis.axes.yaxis.set_ticks(list(range(0, 256, 15)))
    axis.grid(True)
    return figure


def calcula_probabilidade_histograma(hist, num_pixels):
    """."""
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / num_pixels

    return hist_proba


def calcula_probabilidade_acumulada(hist_proba):
    """."""
    probabilidade_acumulada = {}
    soma_probabilidade = 0

    for i in range(0, 256):
        if i == 0:
            pass
        else:
            soma_probabilidade += hist_proba[str(i - 1)]

        probabilidade_acumulada[str(i)] = hist_proba[str(i)] + soma_probabilidade
    return probabilidade_acumulada


def calcular_novo_valor_pixel(probabilidade_acumulada):
    """."""
    novo_valor = {}

    for i in range(0, 256):
        novo_valor[str(i)] = np.ceil(probabilidade_acumulada[str(i)] * 255)
    return novo_valor


def equalizar_histograma(img, novo_valor, isRGB=0, ind=0):
    """."""
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            if isRGB == 0:
                img[row][column] = novo_valor[str(int(img[row][column]))]
            else:
                img[row][column][ind] = novo_valor[str(int(img[row][column][ind]))]
    return img


def processa_equalizacao(imagem, histograma, isRGB=0, ind=0):
    """."""
    num_pixels = imagem.shape[0] * imagem.shape[1]
    hist_proba = calcula_probabilidade_histograma(histograma, num_pixels)
    probabilidade_acumulada = calcula_probabilidade_acumulada(hist_proba)
    novo_valor = calcular_novo_valor_pixel(probabilidade_acumulada)
    imagem = equalizar_histograma(imagem.copy(), novo_valor, isRGB, ind)
    return imagem


def processa_espaco(imagem, histograma, espaco):
    """."""
    from pprint import pprint
    valores_modificar = {}
    if espaco != None:
        for i in espaco:
            for j in range(i[0], i[1] + 1):
                valores_modificar[str(j)] = i[2]
    for row in range(imagem.shape[0]):
        for column in range(imagem.shape[1]):
            if str(int(imagem[row][column])) in valores_modificar.keys():
                imagem[row][column] = valores_modificar[str(int(imagem[row][column]))]
    # num_pixels = imagem.shape[0] * imagem.shape[1]
    # hist_proba = calcula_probabilidade_histograma(histograma, num_pixels)
    # probabilidade_acumulada = calcula_probabilidade_acumulada(hist_proba)
    # novo_valor = calcular_novo_valor_pixel(probabilidade_acumulada)
    # imagem = equalizar_histograma(imagem.copy(), novo_valor, isRGB, ind)
    return imagem


def histograma(imagem, mostraHistograma, equalizar, isRGB, espaco):
    """."""
    if espaco != None:
        isRGB = 0
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        histograma = inicia_histograma()
        histograma = conta_intensidade_pixels(histograma, imagem)
        processa_espaco(imagem, histograma, espaco)
    elif isRGB == 0:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        histograma = inicia_histograma()
        histograma = conta_intensidade_pixels(histograma, imagem)
    else:
        histogramar = inicia_histograma()
        histogramag = inicia_histograma()
        histogramab = inicia_histograma()
        if mostraHistograma in (0, 1):
            histogramar = conta_intensidade_pixels(histogramar, imagem[:, :, 0])
            histogramag = conta_intensidade_pixels(histogramag, imagem[:, :, 1])
            histogramab = conta_intensidade_pixels(histogramab, imagem[:, :, 2])
        if mostraHistograma == 2:
            histogramar = conta_intensidade_pixels(histogramar, imagem[:, :, 0])
        if mostraHistograma == 3:
            histogramag = conta_intensidade_pixels(histogramag, imagem[:, :, 1])
        if mostraHistograma == 4:
            histogramab = conta_intensidade_pixels(histogramab, imagem[:, :, 2])

    if equalizar > 0 and espaco == None:
        if isRGB == 0:
            imagem = processa_equalizacao(imagem, histograma, isRGB)
            if mostraHistograma > 0:
                histograma = inicia_histograma()
                histograma = conta_intensidade_pixels(histograma, imagem)
        else:
            imagem = processa_equalizacao(imagem, histogramar, isRGB, 0)
            imagem = processa_equalizacao(imagem, histogramag, isRGB, 1)
            imagem = processa_equalizacao(imagem, histogramab, isRGB, 2)
            if mostraHistograma > 0:
                histogramar = inicia_histograma()
                histogramag = inicia_histograma()
                histogramab = inicia_histograma()
                if mostraHistograma == 1:
                    histogramar = conta_intensidade_pixels(histogramar, imagem[:, :, 0])
                    histogramag = conta_intensidade_pixels(histogramag, imagem[:, :, 1])
                    histogramab = conta_intensidade_pixels(histogramab, imagem[:, :, 2])
                if mostraHistograma == 2:
                    histogramar = conta_intensidade_pixels(histogramar, imagem[:, :, 0])
                if mostraHistograma == 3:
                    histogramag = conta_intensidade_pixels(histogramag, imagem[:, :, 1])
                if mostraHistograma == 4:
                    histogramab = conta_intensidade_pixels(histogramab, imagem[:, :, 2])

    if mostraHistograma > 0:
        if isRGB == 0:
            f = plot_hist(histograma, espaco)
        else:
            f = plot_hist_rgb(histogramar, histogramag, histogramab)
        output = io.BytesIO()
        FigureCanvas(f).print_png(output)
        img = Image.open(output)
        imagem = np.array(img)
    return imagem
