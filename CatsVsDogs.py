import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 1. BASIC SETTINGS


# All images will be resized to 180x180
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32 #no of images in a batch
EPOCHS = 5  # times to train the dataset

# Folder paths (relative to project root)
train_dir = "data/train"
val_dir = "data/val"



# 2. LOAD DATA FROM FOLDERS


print("Loading training data...")
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Loading validation data...")
val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names #class name taken as labels
print("Class names (folder names):", class_names)

# Optional: cache/prefetch for speed loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# ==========================
# 3. BUILD THE MODEL (CNN)


model = keras.Sequential([
    # Scale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1.0 / 255, input_shape=IMAGE_SIZE + (3,)),

    # Convolution + Pooling block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution + Pooling block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution + Pooling block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # output: 0..1 (cat/dog)
])

print("\nModel summary:")
model.summary()


# ==========================
# 4. COMPILE THE MODEL


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================
# 5. TRAIN THE MODEL


print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\nTraining finished!")


# ==========================
# 6. SAVE THE MODEL


model_path = "cat_dog_model.h5"
model.save(model_path)
print(f"Model saved to: {model_path}")


# ==========================
# 7. PLOT ACCURACY & LOSS


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

# Plot Accuracy
plt.figure()
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Plot Loss
plt.figure()
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# ==========================
# 8. PREDICT ON A NEW IMAGE


def predict_image(image_path):
    """
    Give the path to an image, this will print if it's a CAT or DOG.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    img = keras.utils.load_img(
        image_path, target_size=IMAGE_SIZE
    )
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 180, 180, 3)

    # IMPORTANT: model already has Rescaling layer, so we don't need to divide by 255 again
    predictions = model.predict(img_array)
    score = float(predictions[0][0])

    # Assuming class_names[0] = 'cats', class_names[1] = 'dogs'
    if score < 0.5:
        label = class_names[0]  # "cats"
    else:
        label = class_names[1]  # "dogs"

    print(f"Prediction: {label.upper()}  (score={score:.4f})")


# Example: uncomment & change path to test after training
#put your file path down here to check if your image is working plug and play###################################
predict_image("test_dog.jpg")
