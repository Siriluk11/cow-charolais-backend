import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# =========================
# ตั้งค่า
# =========================
MODEL_DIR = "outputs/models"
DATA_DIR = "dataset/split_dataset/train"   # ใช้ train สำหรับแบ่ง K-Fold
MODEL_NAME = "MobileNetV3Large"            # 🔥 เปลี่ยนชื่อโมเดลได้ตรงนี้
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K = 5

# =========================
# อ่าน path รูปทั้งหมด
# =========================
image_paths = []
labels = []

class_names = sorted(os.listdir(DATA_DIR))

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(class_dir, file))
            labels.append(class_name)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Classes:", class_names)
print("Total images:", len(image_paths))

# =========================
# K-Fold
# =========================
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), start=1):
    print(f"\n===== Fold {fold}/{K} =====")

    model_path = os.path.join(
        MODEL_DIR,
        f"{MODEL_NAME}_fold{fold}_best.keras"
    )

    print("Loading model:", model_path)

    model = load_model(model_path)

    val_paths = image_paths[val_idx]
    val_labels = labels[val_idx]

    df_val = pd.DataFrame({
        "filename": val_paths,
        "class": val_labels
    })

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_generator = datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col="filename",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    y_true = val_generator.classes

    y_pred_prob = model.predict(val_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Fold {fold} Accuracy: {acc:.4f}")
    print(f"Fold {fold} F1-score: {f1:.4f}")

    fold_results.append({
        "Fold": fold,
        "Accuracy": acc,
        "F1-score": f1
    })

# =========================
# สรุปผล
# =========================
results_df = pd.DataFrame(fold_results)

mean_acc = results_df["Accuracy"].mean()
std_acc = results_df["Accuracy"].std()

mean_f1 = results_df["F1-score"].mean()
std_f1 = results_df["F1-score"].std()

summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1-score"],
    "Mean": [mean_acc, mean_f1],
    "Std": [std_acc, std_f1]
})

print("\n===== K-Fold Results =====")
print(results_df)

print("\n===== Summary =====")
print(summary_df)

# =========================
# บันทึกตาราง
# =========================
results_df.to_csv("kfold_results.csv", index=False)
summary_df.to_csv("kfold_summary.csv", index=False)

# =========================
# สร้างกราฟ
# =========================
plt.figure(figsize=(8, 5))

plt.plot(results_df["Fold"], results_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(results_df["Fold"], results_df["F1-score"], marker="s", label="F1-score")

plt.title(f"K-Fold Cross Validation ({MODEL_NAME})")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.xticks(results_df["Fold"])
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("kfold_results.png", dpi=300)
plt.show()