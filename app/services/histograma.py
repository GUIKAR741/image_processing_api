"""."""
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def instantiate_histogram():
    """."""
    hist_array = []

    for i in range(0, 256):
        hist_array.append(str(i))
        hist_array.append(0)

    hist_dct = {hist_array[i]: hist_array[i + 1]
                for i in range(0, len(hist_array), 2)}

    return hist_dct


def count_intensity_values(hist, img):
    """."""
    for row in img:
        for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1

    return hist


def plot_hist(hist):
    """."""
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(hist.keys(), hist.values(), color="red")
    axis.set_xlabel("Níveis intensidade")
    axis.axes.xaxis.set_ticks([])
    axis.grid(True)
    return figure


def get_hist_proba(hist, n_pixels):
    """."""
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels

    return hist_proba


def get_accumulated_proba(hist_proba):
    """."""
    acc_proba = {}
    sum_proba = 0

    for i in range(0, 256):
        if i == 0:
            pass
        else:
            sum_proba += hist_proba[str(i - 1)]

        acc_proba[str(i)] = hist_proba[str(i)] + sum_proba
    return acc_proba


def get_new_gray_value(acc_proba):
    """."""
    new_gray_value = {}

    for i in range(0, 256):
        new_gray_value[str(i)] = np.ceil(acc_proba[str(i)] * 255)
    return new_gray_value


def equalize_hist(img, new_gray_value):
    """."""
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = new_gray_value[str(int(img[row][column]))]

    return img


def histograma(imagem, mostraHistograma, equalizar):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    histogram = instantiate_histogram()
    histogram = count_intensity_values(histogram, imagem)

    if equalizar > 0:
        n_pixels = imagem.shape[0] * imagem.shape[1]
        hist_proba = get_hist_proba(histogram, n_pixels)
        accumulated_proba = get_accumulated_proba(hist_proba)
        new_gray_value = get_new_gray_value(accumulated_proba)
        imagem = equalize_hist(imagem.copy(), new_gray_value)

        histogram = instantiate_histogram()
        histogram = count_intensity_values(histogram, imagem)

    if mostraHistograma > 0:
        f = plot_hist(histogram)
        output = io.BytesIO()
        FigureCanvas(f).print_png(output)
        img = Image.open(output)
        imagem = np.array(img)
    return imagem
