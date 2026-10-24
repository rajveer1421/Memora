"""Memora: AI memory-care platform."""


def create_app():
    from memora.app import create_app as _create
    return _create()
