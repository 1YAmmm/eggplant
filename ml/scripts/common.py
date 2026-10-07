"""Shared na constants at helpers para sa train.py, evaluate.py, at predict.py."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # patahimikin ang TF info logs

import tensorflow as tf
from tensorflow import keras

ML_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ML_DIR / "data" / "dataset"
MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "results"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
MODEL_NAMES = ("mobilenetv2", "efficientnetb0")


def load_split(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    """I-load ang isang split (train/valid/test). Raw 0-255 ang pixels; nasa loob ng model ang preprocessing."""
    ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds


def save_class_names(class_names):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")


def load_class_names():
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def build_model(name: str, num_classes: int, dropout: float = 0.3):
    """Transfer-learning model. Tumatanggap ng RAW 0-255 RGB images (224x224x3),
    kaya hindi na kailangan ng preprocessing sa labas (predict.py, website).
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    if name == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)  # [-1, 1]
    elif name == "efficientnetb0":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        x = inputs  # may sariling rescaling/normalization sa loob ang EfficientNetB0 (0-255)
    else:
        raise ValueError(f"Hindi kilalang model: {name}. Pumili ng isa sa {MODEL_NAMES}")

    base.trainable = False
    x = base(x, training=False)  # training=False: nananatiling inference mode ang BatchNorm
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"eggplant_{name}")


def get_base(model: keras.Model) -> keras.Model:
    """Hanapin ang pretrained base (ang nested Model) sa loob ng model."""
    bases = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if not bases:
        raise RuntimeError("Walang nested base model na nahanap.")
    return bases[0]


def load_trained_model(path):
    return keras.models.load_model(path)
