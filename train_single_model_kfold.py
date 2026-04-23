import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# =========================
# ตั้งค่าเริ่มต้น
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

# =========================
# PATH
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_ROOT = BASE_DIR / "dataset" / "split_dataset"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_DIR = BASE_DIR / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_SAVE_DIR = OUTPUT_DIR / "models"

CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# เลือกโมเดลทีละตัว
# =========================
MODEL_NAME = "EfficientNetB1"
KFOLDS = 5
EPOCHS = 1000                 # เริ่มที่ 400 ก่อน
LEARNING_RATE = 0.0001        # ถ้ายังไม่ดีค่อยเปลี่ยนเป็น 0.0001
BATCH_SIZE = 4
PATIENCE_EARLYSTOP = 15
PATIENCE_REDUCELR = 7

# ถ้ายังไม่ดีค่อยเปลี่ยนเป็น:
# EPOCHS = 1000
# LEARNING_RATE = 0.0001

# หรือ
# EPOCHS = 2000
# LEARNING_RATE = 0.0001

# =========================
# โมเดล
# =========================
MODEL_CONFIGS = {
    "MobileNet": {
        "builder": tf.keras.applications.MobileNet,
        "preprocess": tf.keras.applications.mobilenet.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "MobileNetV2": {
        "builder": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "MobileNetV3Small": {
        "builder": tf.keras.applications.MobileNetV3Small,
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "MobileNetV3Large": {
        "builder": tf.keras.applications.MobileNetV3Large,
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "NASNetMobile": {
        "builder": tf.keras.applications.NASNetMobile,
        "preprocess": tf.keras.applications.nasnet.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "EfficientNetB0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "EfficientNetB1": {
        "builder": tf.keras.applications.EfficientNetB1,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "input_shape": (240, 240, 3),
    },
    "ResNet50V2": {
        "builder": tf.keras.applications.ResNet50V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input,
        "input_shape": (224, 224, 3),
    },
    "InceptionV3": {
        "builder": tf.keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
        "input_shape": (299, 299, 3),
    },
    "Xception": {
        "builder": tf.keras.applications.Xception,
        "preprocess": tf.keras.applications.xception.preprocess_input,
        "input_shape": (299, 299, 3),
    },
}

if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(f"MODEL_NAME ไม่ถูกต้อง: {MODEL_NAME}")

MODEL_CFG = MODEL_CONFIGS[MODEL_NAME]
INPUT_SIZE = MODEL_CFG["input_shape"][:2]
PREPROCESS_FN = MODEL_CFG["preprocess"]

# =========================
# โหลดรายการภาพ
# =========================
def collect_image_paths(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    folder = Path(folder)

    if not folder.exists():
        return items

    for class_dir in sorted(folder.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            for f in class_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    items.append((str(f), class_name))
    return items

train_items = collect_image_paths(TRAIN_DIR)
val_items = collect_image_paths(VAL_DIR)
test_items = collect_image_paths(TEST_DIR)

all_items = train_items + val_items

print(f"Train images: {len(train_items)}")
print(f"Val images:   {len(val_items)}")
print(f"CV total:     {len(all_items)}")
print(f"Test images:  {len(test_items)}")

if len(all_items) == 0:
    raise ValueError("ไม่พบรูปใน train/val กรุณาตรวจ path ให้ถูกต้อง")

if len(test_items) == 0:
    raise ValueError("ไม่พบรูปใน test กรุณาตรวจ path ให้ถูกต้อง")

df = pd.DataFrame(all_items, columns=["filepath", "label"])
test_df = pd.DataFrame(test_items, columns=["filepath", "label"])

print("\nClass distribution (train+val):")
print(df["label"].value_counts())

le = LabelEncoder()
le.fit(pd.concat([df["label"], test_df["label"]], axis=0))

df["label_id"] = le.transform(df["label"])
test_df["label_id"] = le.transform(test_df["label"])

class_names = list(le.classes_)
num_classes = len(class_names)

print("\nClasses:", class_names)
print("num_classes =", num_classes)

# =========================
# Data pipeline
# =========================
AUTOTUNE = tf.data.AUTOTUNE

def decode_and_resize(filename, label, img_size):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)
    return image, label

def make_dataset(filepaths, labels, img_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(
        lambda x, y: decode_and_resize(x, y, img_size),
        num_parallel_calls=AUTOTUNE
    )

    if training:
        ds = ds.shuffle(buffer_size=max(len(filepaths), 1), seed=SEED, reshuffle_each_iteration=True)

    def _prep(image, label):
        image = PREPROCESS_FN(image)
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    ds = ds.map(_prep, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# =========================
# สร้างโมเดล
# =========================
def build_model():
    base_model = MODEL_CFG["builder"](
        weights="imagenet",
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# callbacks
# =========================
def get_callbacks(model_name, fold):
    best_model_path = MODEL_SAVE_DIR / f"{model_name}_fold{fold}_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLYSTOP,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE_REDUCELR,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks, best_model_path

# =========================
# save graph
# =========================
def save_graphs(history, model_name, fold):
    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title(f"{model_name} Fold {fold} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_fold{fold}_accuracy.png", dpi=200)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{model_name} Fold {fold} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_fold{fold}_loss.png", dpi=200)
    plt.close()

# =========================
# K-Fold train
# =========================
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

fold_results = []
history_rows = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df["filepath"], df["label_id"]), start=1):
    print(f"\n🔥 Fold {fold}/{KFOLDS}")

    x_train = df["filepath"].iloc[train_idx].values
    y_train = df["label_id"].iloc[train_idx].values

    x_val = df["filepath"].iloc[val_idx].values
    y_val = df["label_id"].iloc[val_idx].values

    train_ds = make_dataset(x_train, y_train, img_size=INPUT_SIZE, training=True)
    val_ds = make_dataset(x_val, y_val, img_size=INPUT_SIZE, training=False)

    model = build_model()
    callbacks, best_model_path = get_callbacks(MODEL_NAME, fold)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # โหลด best weights ของ fold นั้น
    model = tf.keras.models.load_model(best_model_path)

    # save history rows
    trained_epochs = len(history.history["loss"])
    for epoch_idx in range(trained_epochs):
        history_rows.append({
            "model": MODEL_NAME,
            "fold": fold,
            "epoch": epoch_idx + 1,
            "train_accuracy": history.history["accuracy"][epoch_idx],
            "val_accuracy": history.history["val_accuracy"][epoch_idx],
            "train_loss": history.history["loss"][epoch_idx],
            "val_loss": history.history["val_loss"][epoch_idx],
        })

    # save graphs
    save_graphs(history, MODEL_NAME, fold)

    # evaluate on validation fold
    y_pred = []
    y_true = []

    for x_batch, y_batch in val_ds:
        pred = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(pred, axis=1))
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Fold {fold} F1-score: {f1:.4f}")

    fold_results.append({
        "model": MODEL_NAME,
        "fold": fold,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "epochs_requested": EPOCHS,
        "epochs_trained": trained_epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE
    })

    tf.keras.backend.clear_session()
    gc.collect()

# =========================
# save CV results
# =========================
fold_df = pd.DataFrame(fold_results)
history_df = pd.DataFrame(history_rows)

summary_df = pd.DataFrame([{
    "model": MODEL_NAME,
    "kfold": KFOLDS,
    "epochs_requested": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "accuracy_mean": round(fold_df["accuracy"].mean(), 4),
    "accuracy_std": round(fold_df["accuracy"].std(), 4),
    "f1_score_mean": round(fold_df["f1_score"].mean(), 4),
    "f1_score_std": round(fold_df["f1_score"].std(), 4),
}])

fold_csv = CSV_DIR / f"{MODEL_NAME}_fold_results.csv"
history_csv = CSV_DIR / f"{MODEL_NAME}_history_results.csv"
summary_csv = CSV_DIR / f"{MODEL_NAME}_summary_results.csv"

fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")
history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

print("\n================ FOLD RESULTS ================\n")
print(fold_df.to_string(index=False))

print("\n================ SUMMARY RESULT ================\n")
print(summary_df.to_string(index=False))

# =========================
# FINAL TRAIN ON FULL TRAIN+VAL
# =========================
print("\n🚀 Train final model on full train+val and evaluate on test set")

x_full = df["filepath"].values
y_full = df["label_id"].values

x_test = test_df["filepath"].values
y_test = test_df["label_id"].values

# แบ่งบางส่วนจาก full เป็น validation สำหรับ final training callbacks
# ใช้ 10% ของ full เป็น validation ชั่วคราว
from sklearn.model_selection import train_test_split

x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
    x_full,
    y_full,
    test_size=0.1,
    stratify=y_full,
    random_state=SEED
)

train_final_ds = make_dataset(x_train_final, y_train_final, img_size=INPUT_SIZE, training=True)
val_final_ds = make_dataset(x_val_final, y_val_final, img_size=INPUT_SIZE, training=False)
test_ds = make_dataset(x_test, y_test, img_size=INPUT_SIZE, training=False)

final_model = build_model()
final_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_EARLYSTOP,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=PATIENCE_REDUCELR,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_DIR / f"{MODEL_NAME}_final_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

final_history = final_model.fit(
    train_final_ds,
    validation_data=val_final_ds,
    epochs=EPOCHS,
    callbacks=final_callbacks,
    verbose=1
)

final_model = tf.keras.models.load_model(MODEL_SAVE_DIR / f"{MODEL_NAME}_final_best.keras")

# =========================
# TEST EVALUATION
# =========================
y_test_pred = []
y_test_true = []

for x_batch, y_batch in test_ds:
    pred = final_model.predict(x_batch, verbose=0)
    y_test_pred.extend(np.argmax(pred, axis=1))
    y_test_true.extend(np.argmax(y_batch.numpy(), axis=1))

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average="weighted")

print("\n================ TEST RESULT ================\n")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1-score : {test_f1:.4f}")

report_text = classification_report(
    y_test_true,
    y_test_pred,
    target_names=class_names,
    digits=4
)
print("\nClassification Report:\n")
print(report_text)

test_result_df = pd.DataFrame([{
    "model": MODEL_NAME,
    "test_accuracy": round(test_acc, 4),
    "test_f1_score": round(test_f1, 4),
    "epochs_requested": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE
}])

test_csv = CSV_DIR / f"{MODEL_NAME}_test_results.csv"
test_result_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

with open(CSV_DIR / f"{MODEL_NAME}_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

# final graphs
plt.figure(figsize=(8, 5))
plt.plot(final_history.history["accuracy"], label="train_accuracy")
plt.plot(final_history.history["val_accuracy"], label="val_accuracy")
plt.title(f"{MODEL_NAME} Final Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{MODEL_NAME}_final_accuracy.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(final_history.history["loss"], label="train_loss")
plt.plot(final_history.history["val_loss"], label="val_loss")
plt.title(f"{MODEL_NAME} Final Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{MODEL_NAME}_final_loss.png", dpi=200)
plt.close()

print(f"\n📊 saved fold table   : {fold_csv}")
print(f"📊 saved history table: {history_csv}")
print(f"📊 saved summary table: {summary_csv}")
print(f"📊 saved test table   : {test_csv}")
print(f"📈 saved plots folder : {PLOT_DIR}")
print(f"💾 saved models folder: {MODEL_SAVE_DIR}")

print("\n✅ เสร็จแล้ว")