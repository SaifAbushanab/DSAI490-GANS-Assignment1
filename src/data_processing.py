import os
import tensorflow as tf

from .config import IMG_SIZE, BATCH_SIZE


def load_region_dataset(region, root):
    path = os.path.join(root, region)

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0)

    return ds.prefetch(tf.data.AUTOTUNE)