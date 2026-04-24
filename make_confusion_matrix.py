import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== ตั้งค่า path ======
MODEL_PATH = "outputs/models/MobileNetV3Large_final_best.keras"
TEST_DIR = "dataset/split_dataset/test"

# ====== โหลดโมเดล ======
model = load_model(MODEL_PATH)

# ====== โหลดชุดทดสอบ ======
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ====== ทำนายผล ======
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ====== ค่าจริง ======
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

# ====== สร้าง Confusion Matrix ======
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# ====== บันทึกภาพ ======
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ====== แสดงค่า Precision / Recall / F1-score ======
print(classification_report(y_true, y_pred, target_names=labels))