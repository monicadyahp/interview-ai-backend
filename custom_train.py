import tensorflow as tf
import numpy as np

# Simulasi data kecil agar script bisa dijalankan dengan cepat sebagai bukti
input_shape = (None, 48, 48, 1)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# --- INI ADALAH INTI KRITERIA: Custom Training Loop ---
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # 1. Forward Pass
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    # 2. Backward Pass (Mencari Gradient)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 3. Optimization (Update Bobot)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

print("🚀 Memulai Custom Training Loop (tf.GradientTape)...")
# Jalankan 1 epoch saja sebagai demo
dummy_images = np.random.rand(1, 48, 48, 1).astype(np.float32)
dummy_labels = np.array([1])
current_loss = train_step(dummy_images, dummy_labels)
print(f"✅ Berhasil! Loss saat ini: {current_loss.numpy():.4f}")