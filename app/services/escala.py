"""."""
import numpy as np
import math

# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0


# Padding
def padding(img, H, W, C):
    zimg = np.zeros((H+4, W+4, C))
    zimg[2:H+2, 2:W+2, :C] = img

    # Pad the first/last two col and row
    zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
    zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
    zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
    zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
    zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
    return zimg



def escala(imagem, escalaX, escalaY, tipo):
    """."""
    if escalaY == 1 and escalaX == 1:
        return imagem
    w, h = imagem.shape[:2]
    nova_w, nova_h = int(imagem.shape[0] * escalaX), int(imagem.shape[1] * escalaY)
    imagem_escalada = np.zeros((nova_w, nova_h, imagem.shape[2]))
    for y in range(nova_h):
        for x in range(nova_w):
            pixel = [0]
            if tipo == 1:
                x_nearest = int(np.round(x / escalaX))
                y_nearest = int(np.round(y / escalaY))
                pixel = imagem[x_nearest, y_nearest]
            elif tipo == 2:
                x_ = x/escalaX
                y_ = y/escalaY

                # Finding neighboring points
                x1 = min(int(np.floor(x_)), w-1)
                y1 = min(int(np.floor(y_)), h-1)
                x2 = min(int(np.ceil(x_)), w-1)
                y2 = min(int(np.ceil(y_)), h-1)

                Q11 = np.array(imagem[x1, y1])
                Q12 = np.array(imagem[x2, y1])
                Q21 = np.array(imagem[x1, y2])
                Q22 = np.array(imagem[x2, y2])

                # Interpolating P1 and P2
                P1 = (x2-x_)*Q11 + (x_-x1)*Q12
                P2 = (x2-x_)*Q21 + (x_-x1)*Q22

                if x1 == x2:
                    P1 = Q11
                    P2 = Q22

                # Interpolating P
                pixel = (y2-y_)*P1 + (y_-y1)*P2

                # Rounding P to an int tuple
                pixel = np.round(pixel)
                pixel = tuple(pixel.astype(int))
            imagem_escalada[x, y] = pixel
    return imagem_escalada
