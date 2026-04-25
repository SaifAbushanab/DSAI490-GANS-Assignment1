from src.models import build_autoencoder


def test_build():
    model, _ = build_autoencoder()
    assert model is not None