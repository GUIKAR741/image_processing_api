"""."""
import cv2
import numpy as np


def negative_to_zero(img: np.array) -> np.array:
    """."""
    img = img.copy()
    img[img < 0] = 0
    return img


def get_padding_width_per_side(kernel_size: int) -> int:
    """."""
    # Simple integer division
    return kernel_size // 2


def add_padding_to_image(img: np.array, padding_width: int) -> np.array:
    """."""
    # Array of zeros of shape (img + padding_width)
    img_with_padding = np.zeros(shape=(
        # Multiply with two because we need padding on all sides
        img.shape[0] + padding_width * 2,
        img.shape[1] + padding_width * 2
    ))

    # Change the inner elements
    # For example, if img.shape = (224, 224), and img_with_padding.shape = (226, 226)
    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    img_with_padding[padding_width:-padding_width,
                     padding_width:-padding_width] = img

    return img_with_padding


def convolve(img: np.array, kernel: np.array) -> np.array:
    """."""
    pad = get_padding_width_per_side(len(kernel))
    # To simplify things
    k = kernel.shape[0]

    # 2D array of zeros
    convolved_img = np.zeros(shape=(img.shape[0] - (pad * 2), img.shape[1] - (pad * 2)))
    # Iterate over the rows
    for i in range(convolved_img.shape[0]):
        # Iterate over the columns
        for j in range(convolved_img.shape[1]):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i + k, j:j + k]

            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


def convolucao(imagem, matriz):
    """."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) / 255
    imagem = add_padding_to_image(
        img=imagem,
        padding_width=get_padding_width_per_side(len(matriz))
    )
    imagem = convolve(imagem, matriz)
    # imagem = cv2.filter2D(imagem, -1, matriz)
    return negative_to_zero(imagem) * 255
