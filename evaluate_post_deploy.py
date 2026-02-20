import requests
import os
import mlflow

API_URL = "http://127.0.0.1:8000/predict"
TEST_FOLDER = "test_images"

true_labels = []
pred_labels = []

for filename in os.listdir(TEST_FOLDER):
    if filename.endswith((".jpg", ".png")):

        # infer true label from filename
        if "cat" in filename.lower():
            true_label = "Cat"
        else:
            true_label = "Dog"

        image_path = os.path.join(TEST_FOLDER, filename)

        files = {"file": open(image_path, "rb")}
        response = requests.post(API_URL, files=files)

        result = response.json()

        predicted_label = result["predicted_class"]

        true_labels.append(true_label)
        pred_labels.append(predicted_label)

        print(f"{filename} → Predicted: {predicted_label} | True: {true_label}")

# ---------------------------
# Calculate Post-Deploy Accuracy
# ---------------------------
correct = sum([t == p for t, p in zip(true_labels, pred_labels)])
accuracy = correct / len(true_labels)

print(f"\nPost-Deployment Accuracy: {accuracy:.2f}")

# ---------------------------
# Log to MLflow
# ---------------------------
mlflow.set_experiment("post_deployment_monitoring")

with mlflow.start_run():
    mlflow.log_metric("post_deploy_accuracy", accuracy)

print("Post-deployment accuracy logged to MLflow.")
