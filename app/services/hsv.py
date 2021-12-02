"""."""
import cv2
from math import ceil


def hsv(imagem):
    """."""
    # imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    # return imagem
    (width, height) = imagem.shape[0:2]
    for x in range(width):
        for y in range(height):
            r, g, b = imagem[x, y] / 255
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            v = cmax
            h = 0
            s = 0 if cmax == 0 else delta / cmax
            if delta == 0:
                h = 0
            elif r == cmax:
                h = ((g - b) / delta)
            elif g == cmax:
                h = 2 + (b - r) / delta
            else:
                h = 4 + (r - g) / delta
            h *= 60
            if h < 0:
                h += 360
            imagem[x, y] = [h, s * 255, v * 255]
    return imagem


def rgbTohsv(r, g, b):
    """."""
    r, g, b = r / 255, g / 255, b / 255
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    v = cmax
    h = 0
    s = 0 if cmax == 0 else delta / cmax
    if delta == 0:
        h = 0
    elif r == cmax:
        h = ((g - b) / delta)
    elif g == cmax:
        h = 2 + (b - r) / delta
    else:
        h = 4 + (r - g) / delta
    h *= 60
    if h < 0:
        h += 360
    return {
        'h': ceil(h),
        's': ceil(s * 100),
        'v': ceil(v * 100)
    }
