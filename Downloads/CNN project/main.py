import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1. Load Data from CSV
# ----------------------------
train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

# Features and labels
x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

x_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Normalize to 0–1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# ----------------------------
# 2. Build CNN Model
# ----------------------------
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ----------------------------
# 3. Train Model
# ----------------------------
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# ----------------------------
# 4. Evaluate Model
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("✅ Test accuracy:", test_acc)

# ----------------------------
# 5. Predictions & Visualization
# ----------------------------
predictions = model.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i, :, :, 0], cmap="gray")
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    plt.title(f"Pred: {predicted_label} | True: {true_label}")
    plt.axis("off")
plt.show()

# ----------------------------
# 6. Save One Test Sample as test_image.png
# ----------------------------
sample_img = (x_test[0].reshape(28, 28) * 255).astype("uint8")
Image.fromarray(sample_img).save("test_image.png")
print("💾 Saved sample test image as test_image.png")

# ----------------------------
# 7. Test Custom Image (saved one)
# ----------------------------
image_path = "test_image.png"
image = Image.open(image_path).convert("L").resize((28, 28))
image_arr = np.array(image).astype("float32") / 255.0
image_arr = np.expand_dims(image_arr, (0, -1))  # shape (1,28,28,1)

prediction = model.predict(image_arr)
predicted_label = np.argmax(prediction)
print("🔮 Predicted label from test_image.png:", predicted_label)

# ----------------------------
# 8. Extra: Confusion Matrix & Report
# ----------------------------
y_pred = np.argmax(predictions, axis=1)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
