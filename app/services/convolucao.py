"""."""
import cv2
import numpy as np


def noramalizaImagem(img):
    """."""
    img = img.copy()
    img[img < 0] = 0
    return img


def calculaTamanhoPaddingPorLado(kernel_size):
    """."""
    return kernel_size // 2


def adicionaPaddingNaImagem(img, padding_width):
    """."""
    # Array de zeros nas dimensões da imagem mais padding (img + padding_width)
    img_with_padding = np.zeros(shape=(
        # Multiplica Padding por 2 para adicionar padding em todos lados
        img.shape[0] + padding_width * 2,
        img.shape[1] + padding_width * 2
    ))

    # Adiciona imagem original na imagem com padding
    img_with_padding[padding_width:-padding_width,
                     padding_width:-padding_width] = img

    return img_with_padding


def aplicaConvolucao(img, kernel, funcConv=lambda m, k: np.sum(np.multiply(m, k))):
    """."""
    pad = calculaTamanhoPaddingPorLado(len(kernel))
    # quantidade de Linhas da Imagem
    k = kernel.shape[0]

    # Array do tamanho original da imagem de Zeros
    convolved_img = np.zeros(shape=(img.shape[0] - (pad * 2), img.shape[1] - (pad * 2)))

    for i in range(convolved_img.shape[0]):
        for j in range(convolved_img.shape[1]):
            # Pega a Matriz para calcular a convolução
            mat = img[i:i + k, j:j + k]

            # Aplica a convolução - multiplica elemento a elemento e soma do resultado
            # Armazene o resultado na linha i e coluna j da matriz convolved_img
            convolved_img[i, j] = funcConv(mat, kernel)

    return convolved_img


def convolucao(imagem, matriz, normaliza=0, func=None):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) / 255
    imagem = adicionaPaddingNaImagem(
        img=imagem,
        padding_width=calculaTamanhoPaddingPorLado(len(matriz))
    )
    if func == None:
        imagem = aplicaConvolucao(imagem, matriz)
    else:
        imagem = aplicaConvolucao(imagem, matriz, func)
    return noramalizaImagem(imagem) * 255 if normaliza == 0 else imagem
