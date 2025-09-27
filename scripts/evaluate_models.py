import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from utils import load_datasets

# Path dataset & model
test_dir = "data/test"
resnet_model_path = "models/resnet50_flamevision.keras"
mobilenet_model_path = "models/mobilenetv2_flamevision.keras"

# Load data
_, test_ds, class_names = load_datasets("data/train", test_dir)

# Load models
resnet_model = tf.keras.models.load_model(resnet_model_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)

def evaluate_model(model, test_ds, model_name):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        preds = (preds > 0.5).astype("int32")
        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n{model_name} Evaluation:\n", report)

    with open("results/metrics.txt", "a") as f:
        f.write(f"\n{model_name} Evaluation:\n{report}\n")

# Evaluasi ResNet50
evaluate_model(resnet_model, test_ds, "ResNet50")

# Evaluasi MobileNetV2
evaluate_model(mobilenet_model, test_ds, "MobileNetV2")
