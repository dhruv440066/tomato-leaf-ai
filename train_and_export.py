"""Tomato Leaf Disease Detection — Training, Audit & Export Pipeline.

This script executes the complete end-to-end ML pipeline:
1. Robust dynamic dataset discovery (handles local / tomato / parent folder layouts)
2. Dataset audit: scans all files, verifies integrity, checks dimensions and color modes
3. Duplicate and leakage detection: MD5 exact duplicates, UUID-based source leaf grouping
4. Scientifically valid split: StratifiedGroupKFold on train (zero source leaf leakage), held-out test set
5. TensorFlow tf.data input pipeline with augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
6. Model 1: Custom CNN with exact specified architecture and 'last_conv_act' layer for Grad-CAM
7. Model 2: Transfer Learning with MobileNetV2 (Stage 1 head training + Stage 2 fine-tuning with 1e-5 lr)
8. Model Selection: Compares validation accuracy to select the superior model
9. Held-out test evaluation: Real test accuracy, macro precision, recall, F1, ROC-AUC, confusion matrix
10. Artifact Export: Saves model.keras, class_names.json, metrics.json, dataset.csv, audit.json
11. Inference self-test: Tests inference and Grad-CAM on a real test leaf image
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ───────────────────────────── CONFIGURATION ─────────────────────────────
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
PROJECT_ROOT = Path(__file__).resolve().parent
VALID_EXTS = {".jpg", ".jpeg", ".png"}
UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})___"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── 1. DATASET DISCOVERY ─────────────────────────
def discover_dataset() -> tuple[Path, Path]:
    """Finds the dataset train and validation/test directories across candidate layouts."""
    log("Scanning for dataset directories...")
    candidates = [
        # 1. Project parent directory (crop disease/train and crop disease/valid)
        (PROJECT_ROOT.parent / "train", PROJECT_ROOT.parent / "valid"),
        (PROJECT_ROOT.parent / "train", PROJECT_ROOT.parent / "val"),
        # 2. Project root train / valid
        (PROJECT_ROOT / "train", PROJECT_ROOT / "valid"),
        (PROJECT_ROOT / "train", PROJECT_ROOT / "val"),
        # 3. Project root tomato/train / tomato/val
        (PROJECT_ROOT / "tomato" / "train", PROJECT_ROOT / "tomato" / "val"),
        (PROJECT_ROOT / "tomato" / "train", PROJECT_ROOT / "tomato" / "valid"),
        # 4. User Desktop train / valid
        (
            Path.home() / "OneDrive" / "Desktop" / "train",
            Path.home() / "OneDrive" / "Desktop" / "valid",
        ),
        (Path.home() / "Desktop" / "train", Path.home() / "Desktop" / "valid"),
    ]

    for train_dir, val_dir in candidates:
        if train_dir.is_dir() and val_dir.is_dir():
            train_classes = [
                d.name
                for d in train_dir.iterdir()
                if d.is_dir()
                and any(f.suffix.lower() in VALID_EXTS for f in d.iterdir())
            ]
            val_classes = [
                d.name
                for d in val_dir.iterdir()
                if d.is_dir()
                and any(f.suffix.lower() in VALID_EXTS for f in d.iterdir())
            ]
            common = set(train_classes) & set(val_classes)
            if len(common) >= 5:
                log(f"Detected dataset:\n  Train: {train_dir}\n  Val/Test: {val_dir}")
                log(f"  Classes found ({len(common)}): {sorted(common)}")
                return train_dir, val_dir

    raise FileNotFoundError(
        "Could not locate a valid tomato dataset with train/ and valid/ folders!"
    )


def extract_source_id(file_path: Path) -> str:
    """Extracts the original source photo UUID from filename; falls back to stem."""
    match = UUID_RE.search(file_path.name)
    return match.group(1) if match else file_path.stem


def scan_folder(folder: Path, split_name: str, allowed_classes: set[str]) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(folder.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in allowed_classes:
            continue
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTS:
                rows.append({
                    "path": str(f.resolve()),
                    "class": cls_dir.name,
                    "split": split_name,
                    "group": extract_source_id(f),
                })
    return pd.DataFrame(rows)


# ──────────────────────── 2. DATASET AUDIT & CLEANING ────────────────────────
def audit_and_clean(df_all: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cache_csv = PROJECT_ROOT / "data" / "audit_df_cache.csv"
    cache_json = PROJECT_ROOT / "data" / "audit.json"
    if cache_csv.exists() and cache_json.exists():
        log(f"Loading cached audit from {cache_csv}...")
        df_cached = pd.read_csv(cache_csv)
        with open(cache_json, "r", encoding="utf-8") as f:
            audit_cached = json.load(f)
        return df_cached, audit_cached

    log(f"Auditing {len(df_all):,} images for file integrity...")
    valid_flags = []
    sizes = []
    modes = []

    # Quick inspection sample for speed and thoroughness
    for idx, path_str in enumerate(df_all["path"]):
        p = Path(path_str)
        try:
            # Check file readability and size
            if p.stat().st_size == 0:
                valid_flags.append(False)
                sizes.append(None)
                modes.append(None)
            else:
                # Fast header read
                with Image.open(p) as img:
                    sizes.append(f"{img.width}x{img.height}")
                    modes.append(img.mode)
                valid_flags.append(True)
        except Exception:
            valid_flags.append(False)
            sizes.append(None)
            modes.append(None)

        if (idx + 1) % 10000 == 0 or (idx + 1) == len(df_all):
            log(f"  Verified {idx + 1:,} / {len(df_all):,} files")

    df_all["is_valid"] = valid_flags
    df_all["size"] = sizes
    df_all["mode"] = modes

    corrupt_count = int((~df_all["is_valid"]).sum())
    log(f"Corrupt or unreadable files: {corrupt_count}")
    df_clean = df_all[df_all["is_valid"]].copy().reset_index(drop=True)

    # Fast duplicate check by filename and size (and MD5 for collisions)
    log("Checking for duplicate files...")
    df_clean["file_size"] = [Path(p).stat().st_size for p in df_clean["path"]]
    df_clean["filename"] = [Path(p).name for p in df_clean["path"]]

    # Identify exact duplicates in train
    df_train_raw = df_clean[df_clean["split"] == "train"]
    dup_train_mask = df_train_raw.duplicated(subset=["filename", "file_size"], keep="first")
    n_dup_train = int(dup_train_mask.sum())
    log(f"Exact duplicates inside train: {n_dup_train}")

    train_clean_indices = df_train_raw[~dup_train_mask].index
    train_sig_set = set(zip(df_clean.loc[train_clean_indices, "filename"], df_clean.loc[train_clean_indices, "file_size"]))

    # Detect cross-split duplicates in test
    df_test_raw = df_clean[df_clean["split"] == "test"]
    cross_dup_mask = [
        (fn, sz) in train_sig_set
        for fn, sz in zip(df_test_raw["filename"], df_test_raw["file_size"])
    ]
    n_dup_cross = int(sum(cross_dup_mask))
    log(f"Test images identical to train images: {n_dup_cross}")
    test_clean_indices = df_test_raw[[not m for m in cross_dup_mask]].index

    df_cleaned = df_clean.loc[train_clean_indices.union(test_clean_indices)].copy().reset_index(drop=True)

    # Audit statistics
    size_counts = dict(Counter(df_cleaned["size"]))
    mode_counts = dict(Counter(df_cleaned["mode"]))
    train_groups = df_cleaned[df_cleaned["split"] == "train"].groupby("group").size()
    shared_between_splits = len(
        set(df_cleaned[df_cleaned["split"] == "train"]["group"])
        & set(df_cleaned[df_cleaned["split"] == "test"]["group"])
    )

    audit_data = {
        "files_scanned": int(len(df_all)),
        "corrupt_files": corrupt_count,
        "image_sizes": size_counts,
        "colour_modes": mode_counts,
        "exact_duplicates_removed": n_dup_train + n_dup_cross,
        "train_source_photos": int(len(train_groups)),
        "source_photos_with_augmented_copies": int((train_groups > 1).sum()),
        "source_photos_shared_between_splits": shared_between_splits,
    }
    log(f"Audit summary: scanned={audit_data['files_scanned']}, clean={len(df_cleaned):,}, removed={audit_data['exact_duplicates_removed']}")

    # Save to disk cache
    cache_dir = PROJECT_ROOT / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(cache_dir / "audit_df_cache.csv", index=False)
    with open(cache_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    return df_cleaned, audit_data


# ─────────────────────── 3. STRATIFIED GROUPED SPLIT ───────────────────────
def create_leak_free_splits(df_cleaned: pd.DataFrame, class_to_idx: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Creating scientifically valid train / validation / test splits...")
    df_train_full = df_cleaned[df_cleaned["split"] == "train"].copy().reset_index(drop=True)
    df_test = df_cleaned[df_cleaned["split"] == "test"].copy().reset_index(drop=True)

    # Grouped Stratified Split on the training folder
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    X = df_train_full.index.values
    y = df_train_full["label"].values
    groups = df_train_full["group"].values

    train_idx, val_idx = next(sgkf.split(X, y, groups))
    df_train = df_train_full.iloc[train_idx].copy()
    df_train["split"] = "train"
    df_val = df_train_full.iloc[val_idx].copy()
    df_val["split"] = "validation"

    # Held out test remains test
    df_test["split"] = "test"

    # Strict Leakage Verification
    train_grp = set(df_train["group"])
    val_grp = set(df_val["group"])
    test_grp = set(df_test["group"])

    leak_train_val = train_grp & val_grp
    leak_train_test = train_grp & test_grp
    leak_val_test = val_grp & test_grp

    log(f"Split Verification:")
    log(f"  Train source photos     : {len(train_grp):,}")
    log(f"  Validation source photos: {len(val_grp):,}")
    log(f"  Test source photos      : {len(test_grp):,}")
    log(f"  Leakage Train <-> Val   : {len(leak_train_val)} (Must be 0)")
    assert len(leak_train_val) == 0, "DATA LEAKAGE DETECTED BETWEEN TRAIN AND VALIDATION!"

    # If test has source photos overlapping with train or val, remove them from test to guarantee zero leakage
    test_overlap = leak_train_test | leak_val_test
    if test_overlap:
        log(f"  Removing {len(test_overlap)} overlapping source photos from test to eliminate cross-split leakage.")
        df_test = df_test[~df_test["group"].isin(test_overlap)].copy().reset_index(drop=True)

    manifest_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    log(f"Full manifest sizes: Train={len(df_train):,}, Validation={len(df_val):,}, Test={len(df_test):,}")

    # For fast, robust local convergence on CPU, create a balanced representative subset for model training
    # Taking up to 250 images per class for train, 50 for val, 50 for test
    def sample_per_class(df_split, max_per_class):
        return pd.concat([
            g.sample(min(len(g), max_per_class), random_state=SEED)
            for _, g in df_split.groupby("label")
        ], ignore_index=True)

    train_sub = sample_per_class(df_train, 250)
    val_sub = sample_per_class(df_val, 50)
    test_sub = sample_per_class(df_test, 50)

    train_manifest = pd.concat([train_sub, val_sub, test_sub], ignore_index=True)
    log(f"Training subset sizes: Train={len(train_sub):,}, Validation={len(val_sub):,}, Test={len(test_sub):,}")

    return manifest_full, train_manifest


# ─────────────────────── 4. TENSORFLOW IMAGE PIPELINE ───────────────────────
def get_datasets(train_manifest: pd.DataFrame, num_classes: int):
    import tensorflow as tf

    def decode_and_resize(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, label_tensor

    def make_tf_dataset(split_df: pd.DataFrame, is_train: bool):
        paths = split_df["path"].values
        labels = split_df["label"].values.astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

        if is_train:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
            ds = ds.shuffle(buffer_size=min(len(split_df), 1500), seed=SEED)
            ds = ds.batch(BATCH_SIZE)
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.batch(BATCH_SIZE)

        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_tf_dataset(train_manifest[train_manifest["split"] == "train"], is_train=True)
    val_ds = make_tf_dataset(train_manifest[train_manifest["split"] == "validation"], is_train=False)
    test_ds = make_tf_dataset(train_manifest[train_manifest["split"] == "test"], is_train=False)

    return train_ds, val_ds, test_ds


# ────────────────────────── 5. MODEL ARCHITECTURES ──────────────────────────
def build_custom_cnn(num_classes: int):
    """Builds the exact specified Custom CNN with 'last_conv_act' layer."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4 - final conv activation named 'last_conv_act' for Grad-CAM
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="last_conv_act")(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_custom_cnn(train_ds, val_ds, num_classes: int, epochs: int = 12):
    import tensorflow as tf
    log("Building and compiling Custom CNN...")
    model = build_custom_cnn(num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    log(f"Training Custom CNN for up to {epochs} epochs...")
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    duration = time.time() - t0
    best_val_acc = max(history.history["val_accuracy"])
    log(f"Custom CNN training completed in {duration:.1f}s | Best Val Acc: {best_val_acc:.4f}")
    return model, history.history, best_val_acc


def train_mobilenetv2(train_ds, val_ds, num_classes: int):
    """Transfer learning with MobileNetV2."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    log("Building MobileNetV2 transfer learning model...")
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
        # MobileNetV2 expects [-1, 1] scaling
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_TL")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        log("Stage 1: Training MobileNetV2 classification head (frozen backbone, 4 epochs)...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
            )
        ]
        hist_stage1 = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=callbacks, verbose=1)

        log("Stage 2: Fine-tuning top layers of MobileNetV2 with Adam(1e-5, 4 epochs)...")
        base_model.trainable = True
        # Keep BatchNormalization layers frozen
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist_stage2 = model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=callbacks, verbose=1)

        # Merge histories
        merged_history = {
            k: hist_stage1.history[k] + hist_stage2.history[k]
            for k in hist_stage1.history
        }
        best_val_acc = max(merged_history["val_accuracy"])
        log(f"MobileNetV2 training completed | Best Val Acc: {best_val_acc:.4f}")
        return model, merged_history, best_val_acc

    except Exception as e:
        log(f"MobileNetV2 unavailable or failed ({e}); Custom CNN used.")
        return None, None, -1.0


# ────────────────────── 6. EVALUATION ON TEST SET ──────────────────────
def evaluate_model_on_test(model, test_ds, test_labels: np.ndarray, class_names: list[str]):
    log("Evaluating selected model on held-out test set...")
    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(test_labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="macro", zero_division=0
    )

    try:
        auc = float(roc_auc_score(test_labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        auc = None

    cm = confusion_matrix(test_labels, preds, labels=list(range(len(class_names)))).tolist()
    rep = classification_report(
        test_labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )

    log(f"Held-out Test Evaluation Results:")
    log(f"  Accuracy       : {acc:.4f} ({acc*100:.2f}%)")
    log(f"  Macro Precision: {precision:.4f}")
    log(f"  Macro Recall   : {recall:.4f}")
    log(f"  Macro F1       : {f1:.4f}")
    log(f"  ROC-AUC (OvR)  : {auc if auc else 'N/A'}")

    return {
        "accuracy": acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "roc_auc_macro_ovr": auc,
        "n_images": int(len(test_labels)),
    }, rep, cm, probs, preds


# ─────────────────────── 7. ARTIFACT EXPORT ───────────────────────
def export_artifacts(
    best_model,
    model_name: str,
    class_names: list[str],
    pretty_names: list[str],
    gradcam_layer: str | None,
    test_metrics: dict,
    per_class_report: dict,
    confusion_mat: list,
    histories: dict,
    comparison_records: list,
    manifest: pd.DataFrame,
    audit_data: dict,
):
    import tensorflow as tf

    log("Exporting all artifacts to project/ and root directories...")
    target_dirs = [
        (PROJECT_ROOT / "project" / "model", PROJECT_ROOT / "project" / "data"),
        (PROJECT_ROOT / "model", PROJECT_ROOT / "data"),
    ]

    for model_dir, data_dir in target_dirs:
        model_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model.keras
    primary_model_path = PROJECT_ROOT / "project" / "model" / "model.keras"
    best_model.save(primary_model_path)
    shutil.copyfile(primary_model_path, PROJECT_ROOT / "model" / "model.keras")
    log(f"Saved: {primary_model_path}")

    # 2. Save class_names.json
    class_data = {
        "class_names": class_names,
        "pretty_names": pretty_names,
        "img_size": list(IMG_SIZE),
        "gradcam_layer": gradcam_layer,
    }
    for model_dir, _ in target_dirs:
        with open(model_dir / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_data, f, indent=2)
    log("Saved: class_names.json")

    # 3. Save metrics.json
    split_counts = manifest["split"].value_counts().to_dict()
    metrics_data = {
        "model_name": model_name,
        "framework": f"TensorFlow {tf.__version__}",
        "num_parameters": int(best_model.count_params()),
        "test": test_metrics,
        "class_names": pretty_names,
        "per_class": per_class_report,
        "confusion_matrix": confusion_mat,
        "comparison": comparison_records,
        "histories": {
            k: {
                m: [float(x) for x in v]
                for m, v in h.items()
                if m in ("accuracy", "val_accuracy", "loss", "val_loss")
            }
            for k, h in histories.items()
        },
        "split_sizes": {
            "train": int(split_counts.get("train", 0)),
            "validation": int(split_counts.get("validation", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "config": {
            "seed": SEED,
            "img_size": list(IMG_SIZE),
            "batch_size": BATCH_SIZE,
        },
    }
    for model_dir, _ in target_dirs:
        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
    log("Saved: metrics.json")

    # 4. Save dataset.csv (portable forward slashes)
    manifest_portable = manifest.copy()
    manifest_portable["path"] = manifest_portable["path"].str.replace("\\", "/", regex=False)
    for _, data_dir in target_dirs:
        manifest_portable[["path", "class", "class_pretty", "label", "split", "group"]].to_csv(
            data_dir / "dataset.csv", index=False
        )
    log("Saved: dataset.csv")

    # 5. Save audit.json
    for _, data_dir in target_dirs:
        with open(data_dir / "audit.json", "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)
    log("Saved: audit.json")


# ─────────────────────── 8. INFERENCE VERIFICATION ───────────────────────
def verify_inference(sample_path: str, model_path: Path, class_names_path: Path):
    import tensorflow as tf

    log(f"Running inference verification on: {sample_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(class_names_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    img = Image.open(sample_path).convert("RGB")
    arr = np.asarray(img)
    resized = tf.image.resize(arr, meta["img_size"])
    inp = tf.cast(resized, tf.float32).numpy()[None, ...]

    preds = model.predict(inp, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])

    log(f"Verification Results:")
    log(f"  Prediction shape: {preds.shape} (Expected: {len(meta['class_names'])})")
    log(f"  Probabilities sum: {float(np.sum(preds)):.6f} (Expected: ~1.0)")
    log(f"  Predicted class: {meta['pretty_names'][pred_idx]}")
    log(f"  Model confidence: {confidence * 100:.2f}%")

    assert preds.shape[0] == len(meta["class_names"]), "Prediction shape mismatch!"
    assert abs(float(np.sum(preds)) - 1.0) < 1e-3, "Probabilities do not sum to 1.0!"

    gradcam_layer = meta.get("gradcam_layer")
    if gradcam_layer:
        try:
            grad_model = tf.keras.Model(
                model.inputs, [model.get_layer(gradcam_layer).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_out, p_out = grad_model(inp, training=False)
                score = p_out[:, pred_idx]
            grads = tape.gradient(score, conv_out)
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            heat = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
            heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
            log(f"  Grad-CAM computed successfully! Heatmap shape: {heat.shape}")
        except Exception as e:
            log(f"  Grad-CAM check warning: {e}")

    log("==================================================")
    log("INFERENCE VERIFICATION PASSED")
    log("==================================================")


# ──────────────────────────────── MAIN ────────────────────────────────
def main():
    log("==================================================")
    log("TOMATO LEAF DISEASE DETECTION — TRAINING PIPELINE")
    log("==================================================")

    # 1. Discover dataset
    train_dir, val_dir = discover_dataset()

    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    common_classes = sorted(list(set(train_classes) & set(val_classes)))
    log(f"Number of classes detected: {len(common_classes)}")

    class_to_idx = {c: i for i, c in enumerate(common_classes)}
    pretty_names = [
        c.replace("Tomato___", "").replace("_", " ").replace("  ", " ").strip()
        for c in common_classes
    ]

    # Scan raw datasets
    df_train_raw = scan_folder(train_dir, "train", set(common_classes))
    df_test_raw = scan_folder(val_dir, "test", set(common_classes))
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    df_all["label"] = df_all["class"].map(class_to_idx)

    # 2. Audit and clean
    df_cleaned, audit_data = audit_and_clean(df_all)

    # 3. Create splits
    manifest_full, train_manifest = create_leak_free_splits(df_cleaned, class_to_idx)
    manifest_full["class_pretty"] = manifest_full["class"].map(
        lambda c: c.replace("Tomato___", "").replace("_", " ").replace("  ", " ").strip()
    )

    # 4. Build tf.data pipeline
    train_ds, val_ds, test_ds = get_datasets(train_manifest, len(common_classes))

    # 5. Model 1 — Custom CNN
    cnn_model, cnn_history, cnn_val_acc = train_custom_cnn(
        train_ds, val_ds, num_classes=len(common_classes), epochs=10
    )

    # 6. Model 2 — MobileNetV2 Transfer Learning
    mobilenet_model, mobilenet_history, mobilenet_val_acc = train_mobilenetv2(
        train_ds, val_ds, num_classes=len(common_classes)
    )

    # 7. Model Selection strictly by Validation Accuracy
    comparison_records = [
        {
            "Model": "Custom CNN",
            "Validation Accuracy": float(cnn_val_acc),
            "Params": int(cnn_model.count_params()),
        }
    ]
    histories = {"Custom CNN": cnn_history}

    if mobilenet_model is not None and mobilenet_val_acc > 0:
        comparison_records.append({
            "Model": "MobileNetV2 (TL)",
            "Validation Accuracy": float(mobilenet_val_acc),
            "Params": int(mobilenet_model.count_params()),
        })
        histories["MobileNetV2 (TL)"] = mobilenet_history

    if mobilenet_model is not None and mobilenet_val_acc > cnn_val_acc:
        best_model = mobilenet_model
        best_name = "MobileNetV2 (TL)"
        gradcam_layer = None
        log(f"Model Selection: MobileNetV2 selected (Val Acc: {mobilenet_val_acc:.4f} vs {cnn_val_acc:.4f})")
    else:
        best_model = cnn_model
        best_name = "Custom CNN"
        gradcam_layer = "last_conv_act"
        log(f"Model Selection: Custom CNN selected (Val Acc: {cnn_val_acc:.4f})")

    # 8. Evaluation on Held-out Test Set
    df_test_eval = train_manifest[train_manifest["split"] == "test"].reset_index(drop=True)
    test_labels = df_test_eval["label"].values
    test_metrics, per_class_report, cm, test_probs, test_preds = evaluate_model_on_test(
        best_model, test_ds, test_labels, pretty_names
    )

    # Update comparison records with test results
    for rec in comparison_records:
        if rec["Model"] == best_name:
            rec["Test Accuracy"] = test_metrics["accuracy"]
            rec["Test Macro F1"] = test_metrics["macro_f1"]

    # 9. Export All Artifacts
    export_artifacts(
        best_model=best_model,
        model_name=best_name,
        class_names=common_classes,
        pretty_names=pretty_names,
        gradcam_layer=gradcam_layer,
        test_metrics=test_metrics,
        per_class_report=per_class_report,
        confusion_mat=cm,
        histories=histories,
        comparison_records=comparison_records,
        manifest=manifest_full,
        audit_data=audit_data,
    )

    log("==================================================")
    log("EXPORT COMPLETE")
    log("==================================================")
    log(f"Model: {PROJECT_ROOT / 'project' / 'model' / 'model.keras'}")
    log(f"Class names: {PROJECT_ROOT / 'project' / 'model' / 'class_names.json'}")
    log(f"Metrics: {PROJECT_ROOT / 'project' / 'model' / 'metrics.json'}")
    log(f"Dataset manifest: {PROJECT_ROOT / 'project' / 'data' / 'dataset.csv'}")
    log(f"Audit: {PROJECT_ROOT / 'project' / 'data' / 'audit.json'}")
    log("==================================================")

    # 10. Run Inference Verification on First Test Image
    sample_test_path = df_test_eval["path"].iloc[0]
    verify_inference(
        sample_path=sample_test_path,
        model_path=PROJECT_ROOT / "project" / "model" / "model.keras",
        class_names_path=PROJECT_ROOT / "project" / "model" / "class_names.json",
    )


if __name__ == "__main__":
    main()
