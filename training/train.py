import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Dataset path
DATASET_PATH = "dataset/PlantVillage/PlantVillage"

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Class names
class_names = train_dataset.class_names

print("\nDisease Classes:")
for i, name in enumerate(class_names):
    print(i, ":", name)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

# CNN model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model
model.summary()

# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save trained model
model.save("model/plant_disease_model.keras")

# Save class names
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n✅ Training completed!")
print("✅ Model saved successfully.")