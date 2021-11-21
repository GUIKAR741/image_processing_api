"""Modulo Principal do Projeto."""
from decouple import config
from flask import Flask


def start_app() -> Flask:
    """Inicia o App."""
    app = Flask(__name__)

    app.config.from_object(
        "app.config." + config('FLASK_ENV', default='production', cast=str)
    )

    @app.errorhandler(404)
    def page_not_found(e):  # pylint: disable=unused-variable
        """Error Page Not Found."""
        return {"status": "rota não encontrada!"}, 404

    @app.errorhandler(500)
    def server_error(e):  # pylint: disable=unused-variable
        """Error Internal Server Error."""
        return {"status": "erro no servidor!"}, 500

    @app.errorhandler(401)
    def unauthorized(e):  # pylint: disable=unused-variable
        """Error unauthorized."""
        return {"status": "erro não autorizado!"}, 401

    @app.errorhandler(403)
    def forbidden(e):  # pylint: disable=unused-variable
        """Error unauthorized."""
        return {"status": "Você não possui permissão para acessar!"}, 403

    from .controllers import index
    app.register_blueprint(index)

    return app
