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


def load_split(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = False, img_size: int = IMG_SIZE):
    """I-load ang isang split (train/valid/test). Raw 0-255 ang pixels; nasa loob ng model ang preprocessing."""
    ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        label_mode="int",
        image_size=(img_size, img_size),
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


def check_class_count(model: keras.Model, class_names):
    """Hihinto kapag hindi tugma ang output ng model sa class_names.json (hal. lumang 5-class model, 6 na ang classes)."""
    n = int(model.output_shape[-1])
    if n != len(class_names):
        raise SystemExit(f"{n} classes ang model pero {len(class_names)} ang nasa {CLASS_NAMES_PATH.name} "
                         f"({class_names}). Ibang dataset ang ginamit sa model na ito.")


def build_model(name: str, num_classes: int, dropout: float = 0.3, img_size: int = IMG_SIZE):
    """Transfer-learning model. Tumatanggap ng RAW 0-255 RGB images (img_size x img_size x 3),
    kaya hindi na kailangan ng preprocessing sa labas (predict.py, website).
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    if name == "mobilenetv2":
        base = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)  # [-1, 1]
    elif name == "efficientnetb0":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
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


def model_img_size(model: keras.Model) -> int:
    """Laki ng input (pixels) ng model; para magamit ng evaluate.py at predict.py ang tamang resize."""
    return int(model.input_shape[1])
