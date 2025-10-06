import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from utils import load_datasets

# Path dataset & model
test_dir = "data/test"
resnet_model_path = "models/leakyrelu_softmax/resnet50_flamevision.keras"
mobilenet_model_path = "models/leakyrelu_softmax/mobilenetv2_flamevision.keras"

# Load data
_, test_ds, class_names = load_datasets("data/train", test_dir, label_mode="categorical")

# Load models
resnet_model = tf.keras.models.load_model(resnet_model_path)
mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)

def evaluate_model(model, test_ds, model_name):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(labels, axis=1)
        y_true.extend(labels)
        y_pred.extend(preds.flatten())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n{model_name} Evaluation:\n", report)

    with open("results/metrics_leakyrelu_softmax.txt", "a") as f:
        f.write(f"\n{model_name} Evaluation:\n{report}\n")

# Evaluasi MobileNetV2
evaluate_model(mobilenet_model, test_ds, "MobileNetV2")

# Evaluasi ResNet50
evaluate_model(resnet_model, test_ds, "ResNet50")
