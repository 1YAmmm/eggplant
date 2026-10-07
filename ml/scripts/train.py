"""Sprint 2 + 3: training (baseline, tapos fine-tuning).

Stage 1 (baseline): naka-freeze ang pretrained base, top layer lang ang tinatrain.
Stage 2 (fine-tune): i-unfreeze ang huling N layers (BatchNorm ay naka-freeze pa rin), mababang learning rate.

Mga output:
  ml/models/<model>_baseline.keras     best model pagkatapos ng stage 1
  ml/models/<model>_finetuned.keras    best model pagkatapos ng stage 2
  ml/models/class_names.json
  ml/results/<model>_history.png       accuracy at loss graphs
  ml/results/<model>_history.json

Usage:
  python ml/scripts/train.py --model mobilenetv2                  (stage 1 at stage 2)
  python ml/scripts/train.py --model mobilenetv2 --fine-tune-epochs 0   (baseline lang)
  python ml/scripts/train.py --model efficientnetb0
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import common
from common import keras, tf


def make_augmenter():
    """Flip, rotation, zoom, contrast. Sa training data lang ginagamit, at sa memory lang (walang naka-save sa disk)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def prepare(train_ds, valid_ds):
    augment = make_augmenter()
    autotune = tf.data.AUTOTUNE
    train = (
        train_ds.cache()
        .shuffle(1000, seed=common.SEED)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    valid = valid_ds.cache().prefetch(autotune)
    return train, valid


def class_weights_from(train_ds, num_classes):
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return {i: float(w) for i, w in enumerate(weights)}


def callbacks_for(ckpt_path, patience):
    return [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
    ]


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(model, n_layers):
    """I-unfreeze ang huling n_layers ng base. Naka-freeze pa rin ang lahat ng BatchNormalization."""
    base = common.get_base(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    trainable = sum(1 for layer in base.layers if layer.trainable)
    print(f"Fine-tuning: {trainable} trainable layers sa base (ng {len(base.layers)})")


def plot_history(history, finetune_start, out_path, title):
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        if finetune_start:
            ax.axvline(finetune_start + 0.5, color="gray", linestyle="--", label="start ng fine-tuning")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{title}: {label}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=common.MODEL_NAMES, default="mobilenetv2")
    ap.add_argument("--epochs", type=int, default=15, help="epochs ng stage 1 (baseline)")
    ap.add_argument("--fine-tune-epochs", type=int, default=10, help="epochs ng stage 2; 0 para i-skip")
    ap.add_argument("--fine-tune-layers", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine-tune-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    args = ap.parse_args()

    keras.utils.set_random_seed(common.SEED)
    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_train = common.load_split("train", args.batch_size, shuffle=True)
    raw_valid = common.load_split("valid", args.batch_size)
    class_names = raw_train.class_names
    assert class_names == raw_valid.class_names, "Hindi magkapareho ang class order ng train at valid"
    common.save_class_names(class_names)
    print("Classes:", class_names)

    weights = class_weights_from(raw_train, len(class_names))
    print("Class weights:", weights)
    train_ds, valid_ds = prepare(raw_train, raw_valid)

    model = common.build_model(args.model, len(class_names), args.dropout)
    model.summary(line_length=100)

    # ---- Stage 1: baseline (naka-freeze ang base)
    baseline_path = common.MODELS_DIR / f"{args.model}_baseline.keras"
    compile_model(model, args.lr)
    t0 = time.time()
    h1 = model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs, class_weight=weights,
        callbacks=callbacks_for(baseline_path, patience=5),
    )
    print(f"Stage 1 tapos: {(time.time() - t0) / 60:.1f} min")
    history = {k: list(map(float, v)) for k, v in h1.history.items() if k in ("accuracy", "loss", "val_accuracy", "val_loss")}
    stage1_epochs = len(history["accuracy"])

    # ang EarlyStopping ay nag-restore ng best weights; i-save ulit para siguradong ito ang baseline file
    model.save(baseline_path)

    # ---- Stage 2: fine-tuning
    if args.fine_tune_epochs > 0:
        finetuned_path = common.MODELS_DIR / f"{args.model}_finetuned.keras"
        unfreeze_top_layers(model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_lr)  # kailangang i-compile ulit pagkatapos baguhin ang trainable
        t0 = time.time()
        h2 = model.fit(
            train_ds, validation_data=valid_ds, epochs=args.fine_tune_epochs, class_weight=weights,
            callbacks=callbacks_for(finetuned_path, patience=4),
        )
        print(f"Stage 2 tapos: {(time.time() - t0) / 60:.1f} min")
        for k in history:
            history[k] += list(map(float, h2.history[k]))
        model.save(finetuned_path)

    (common.RESULTS_DIR / f"{args.model}_history.json").write_text(
        json.dumps({"stage1_epochs": stage1_epochs, **history}, indent=2), encoding="utf-8"
    )
    plot_history(
        history,
        stage1_epochs if args.fine_tune_epochs > 0 else None,
        common.RESULTS_DIR / f"{args.model}_history.png",
        args.model,
    )
    print(f"Final train acc {history['accuracy'][-1]:.3f} | val acc {history['val_accuracy'][-1]:.3f}")
    print("Susunod: python ml/scripts/evaluate.py --model-path ml/models/" + f"{args.model}_baseline.keras")


if __name__ == "__main__":
    main()
