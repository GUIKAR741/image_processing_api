"""."""
from flask import (
    Blueprint,
    request,
    jsonify,
    Response,
    make_response
)
from json import loads
import numpy as np
import cv2
from werkzeug.datastructures import FileStorage
from ..util.reqparser import RequestParser
from ..services.grayscalePonderado import grayscalePonderado
from ..services.grayscaleMedia import grayscaleMedia
from ..services.negativo import negativo
from ..services.esc_log import escalaLogaritmica
from ..services.gamma_correction import gammaCorrection
from ..services.convolucao import convolucao


index = Blueprint("init", __name__)


@index.route('/api/grayscalePonderado', methods=['POST'])
def grayscalePonderadoRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    p = parser.parse_args()
    imagem = p['imagem']
    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = grayscalePonderado(img)

    _, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/grayscaleMedia', methods=['POST'])
def grayscaleMediaRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    p = parser.parse_args()
    imagem = p['imagem']
    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = grayscaleMedia(img)

    _, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/negativo', methods=['POST'])
def negativoRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    p = parser.parse_args()
    imagem = p['imagem']
    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = negativo(img)

    _, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/log', methods=['POST'])
def escalaLogaritmicaRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    parser.add_argument(
        "contrast",
        type=float,
        default=1
    )
    p = parser.parse_args()
    imagem = p['imagem']
    contrast = p["contrast"]
    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = escalaLogaritmica(img, contrast)

    _, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/gamma', methods=['POST'])
def gammaCorrectionRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    parser.add_argument(
        "gamma",
        type=float,
        default=1
    )
    parser.add_argument(
        "contrast",
        type=float,
        default=1
    )
    p = parser.parse_args()
    imagem = p['imagem']
    gamma = p["gamma"]
    contrast = p["contrast"]

    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = gammaCorrection(img, gamma, contrast)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/convolucao', methods=['POST'])
def convolucaoRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    parser.add_argument(
        "tamanho",
        type=int,
        default=1,
        required=True
    )
    parser.add_argument(
        "matriz",
    )
    p = parser.parse_args()
    imagem = p['imagem']
    tamanho = p["tamanho"]
    matriz = loads(p["matriz"])
    matriz = np.array([matriz[i:i + tamanho] for i in range(0, tamanho * tamanho, tamanho)])
    if tamanho % 2 == 0:
        return jsonify({
            "status": "Tamanho deve ser impar!",
        })

    # convert string of image data to uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = convolucao(img, matriz)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response
