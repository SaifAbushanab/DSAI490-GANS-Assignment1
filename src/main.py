import os

from .config import (
    CLASSES,
    DATASET_PATH,
    MODEL_DIR
)

from .data_processing import load_region_dataset
from .trainer import train_autoencoder


def main():

    for region in CLASSES:

        print("Training", region)

        dataset = load_region_dataset(
            region,
            DATASET_PATH
        )

        save_dir = os.path.join(
            MODEL_DIR,
            region
        )

        train_autoencoder(
            dataset,
            save_dir
        )


if __name__ == "__main__":
    main()