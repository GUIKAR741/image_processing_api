"""Arquivo de Configurações."""


class Config:
    """Classe Base das Configurações."""


class development(Config):
    """Configurações de Desenvolvimento."""

    DEBUG = True


class production(Config):
    """Configurações de Produção."""

    DEBUG = False
