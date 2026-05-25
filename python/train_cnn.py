
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tf2onnx
import onnx

# 1. Load Data
print("Loading 'gestures.csv'...")
data = pd.read_csv('gestures.csv')
X = data.drop('label', axis=1).values.reshape(-1, 21, 3)
y = data['label'].values

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(21, 3)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Training...")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# --- NEW: CALCULATE AND PRINT STATISTICS ---
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print("\n--- Training Statistics ---")
print(f"Average Training Accuracy: {np.mean(train_acc)*100:.2f}%")
print(f"Average Validation Accuracy: {np.mean(val_acc)*100:.2f}%")
print(f"Average Training Loss: {np.mean(train_loss):.4f}")
print(f"Average Validation Loss: {np.mean(val_loss):.4f}")

# Find best epoch (highest validation accuracy)
best_epoch = np.argmax(val_acc)
print(f"\nBest Epoch: {best_epoch + 1}")
print(f"  - Val Accuracy: {val_acc[best_epoch]*100:.2f}%")
print(f"  - Val Loss: {val_loss[best_epoch]:.4f}")
# -------------------------------------------

# 4. Generate Accuracy/Loss Graphs
plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.legend()
# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_results.png')
print("Graphs saved to 'training_results.png'")

# 5. Export to ONNX
print("Exporting ONNX...")
spec = (tf.TensorSpec((None, 21, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save_model(onnx_model, "gesture_cnn.onnx")
print("DONE: 'gesture_cnn.onnx' is ready.")
