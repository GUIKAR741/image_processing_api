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
from ..services.blackAndWhite import blackAndWhite
from ..services.negativo import negativo
from ..services.hsv import hsv, rgbTohsv
from ..services.esc_log import escalaLogaritmica
from ..services.gamma_correction import gammaCorrection
from ..services.convolucao import convolucao
from ..services.histograma import histograma
from ..services.sepia import sepia
from ..services.chromakey import chromakey
from ..services.rotacao import rotacao
from ..services.escala import escala
from ..services.laplaciano import laplaciano
from ..services.median import median
from ..services.estenografia import (
    estenografiaLSB,
    estenografiaLSBDecrypt,
    estenografia,
    estenografiaDecrypt
)


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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = grayscalePonderado(img)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = grayscaleMedia(img)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/blackAndWhite', methods=['POST'])
def blackAndWhiteRoute():
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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = blackAndWhite(img)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/sepia', methods=['POST'])
def sepiaRoute():
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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = sepia(img)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/chromakey', methods=['POST'])
def chromakeyRoute():
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
        "r",
        type=int,
        help='Cor Vermelho deve ser enviada!',
        required=True
    )
    parser.add_argument(
        "g",
        type=int,
        help='Cor Verde deve ser enviada!',
        required=True
    )
    parser.add_argument(
        "b",
        type=int,
        help='Cor Azul deve ser enviada!',
        required=True
    )
    parser.add_argument(
        "d",
        type=int,
        help='Distancia deve ser enviada!',
        required=True
    )
    p = parser.parse_args()
    imagem = p['imagem']
    r = p['r']
    g = p['g']
    b = p['b']
    d = p['d']
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = chromakey(img, r, g, b, d)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
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
    parser.add_argument(
        "isRGB",
        type=int,
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    isRGB = p['isRGB']
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = negativo(img, isRGB)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/hsv', methods=['POST'])
def hsvRoute():
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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = hsv(img)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/rgb2hsv', methods=['POST'])
def rgb2hsvRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "r",
        type=int,
        help='Arquivo deve ser enviado!',
        required=True
    )
    parser.add_argument(
        "g",
        type=int,
        help='Arquivo deve ser enviado!',
        required=True
    )
    parser.add_argument(
        "b",
        type=int,
        help='Arquivo deve ser enviado!',
        required=True
    )
    p = parser.parse_args()
    r = p['r']
    g = p['g']
    b = p['b']

    return jsonify(
        rgbTohsv(r, g, b)
    )


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
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = escalaLogaritmica(img, contrast)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
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

    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
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

    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = convolucao(img, matriz)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/histograma', methods=['POST'])
def histogramaRoute():
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
        "mostraHistograma",
        type=int,
        default=0
    )
    parser.add_argument(
        "equalizar",
        type=int,
        default=0
    )
    parser.add_argument(
        "isRGB",
        type=int,
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    mostraHistograma = p['mostraHistograma']
    equalizar = p['equalizar']
    isRGB = p['isRGB']

    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = histograma(img, mostraHistograma, equalizar, isRGB)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/estenografia', methods=['POST'])
def estenografiaRoute():
    """."""
    parser = RequestParser()
    parser.add_argument(
        "imagem1",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    parser.add_argument(
        "imagem2",
        type=FileStorage,
        help='Arquivo deve ser enviado!',
        location='files',
        required=True
    )
    parser.add_argument(
        "tipo",
        type=int,
        default=1,
        required=True
    )
    p = parser.parse_args()
    imagem1 = p['imagem1']
    imagem2 = p['imagem2']
    tipo = p["tipo"]

    # converte string de dados da imagem para uint8
    nparr1 = np.fromstring(imagem1.read(), np.uint8)
    nparr2 = np.fromstring(imagem2.read(), np.uint8)
    # decodifica imagem
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    if tipo == 1:
        img = estenografiaLSB(img1, img2)
    elif tipo == 2:
        img = estenografiaLSBDecrypt(img1)
    elif tipo == 3:
        img = estenografia(img1, img2)
    elif tipo == 4:
        img = estenografiaDecrypt(img1)

    _, buffer = cv2.imencode(f'.{imagem1.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem1.mimetype
    return response


@index.route('/api/rotacao', methods=['POST'])
def rotacaoRoute():
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
        "angulo",
        type=int,
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    angulo = p['angulo']
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = rotacao(img, angulo)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/escala', methods=['POST'])
def escalaRoute():
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
        "tipo",
        type=int,
        default=0
    )
    parser.add_argument(
        "escalaX",
        type=float,
        default=0
    )
    parser.add_argument(
        "escalaY",
        type=float,
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    tipo = p['tipo']
    escalaX = p['escalaX']
    escalaY = p['escalaY']
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = escala(img, escalaX, escalaY, tipo)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/laplaciano', methods=['POST'])
def laplacianoRoute():
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
        "bordas",
        type=int,
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    bordas = p['bordas']
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = laplaciano(img, bordas)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response


@index.route('/api/median', methods=['POST'])
def medianRoute():
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
        default=0
    )
    p = parser.parse_args()
    imagem = p['imagem']
    tamanho = p['tamanho']
    if tamanho % 2 == 0:
        return jsonify({
            "status": "Tamanho deve ser impar!",
        })
    # converte string de dados da imagem para uint8
    nparr = np.fromstring(imagem.read(), np.uint8)
    # decodifica imagem
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = median(img, tamanho)

    _, buffer = cv2.imencode(f'.{imagem.filename.split(".")[-1]}', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = imagem.mimetype
    return response
