import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Input

# --- KRITERIA 1: FUNCTIONAL API (Bukan Sequential) ---
def build_model_functional():
    inputs = Input(shape=(48, 48, 1)) # Layer Input
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(7, activation='softmax')(x)
    
    # Menghubungkan input dan output
    return models.Model(inputs=inputs, outputs=outputs)

model = build_model_functional()
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# --- KRITERIA 6: Custom Training Loop (GradientTape) ---
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

print("🚀 Memulai Custom Training Loop (tf.GradientTape)...")
# DUMMY TEST (Punyamu yang ini bagus, pertahankan!)
dummy_images = np.random.rand(1, 48, 48, 1).astype(np.float32)
dummy_labels = np.array([1])
current_loss = train_step(dummy_images, dummy_labels)
print(f"✅ Berhasil! Loss saat ini: {current_loss.numpy():.4f}")