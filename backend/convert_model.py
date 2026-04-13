import tensorflow as tf

model = tf.keras.models.load_model("MobileNetV3Large_final_best.keras", compile=False)
model.save("model.h5")

print("✅ convert to h5 success")