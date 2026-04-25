import os

from .models import build_autoencoder
from .visualize import plot_loss
from .config import EPOCHS


def train_autoencoder(dataset, save_dir):

    model, encoder = build_autoencoder()

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    history = model.fit(
        dataset.map(lambda x: (x, x)),
        epochs=EPOCHS
    )

    os.makedirs(save_dir, exist_ok=True)

    model.save(
        os.path.join(save_dir, "ae.keras")
    )

    plot_loss(history.history["loss"], "AE Loss")

    return model, encoder