import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from utils import load_datasets, plot_history
os.makedirs("models", exist_ok=True)

# Path dataset
train_dir = "data/train"
test_dir = "data/test"

# Load data
train_ds, test_ds, _ = load_datasets(train_dir, test_dir)

# Model ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layer awal

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=10)

# Simpan model & grafik
model.save("./models/resnet50_flamevision.keras")
plot_history(history, "ResNet50 Training", "./results/resnet50_history.png")
