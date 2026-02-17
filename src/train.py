# import torch
# import mlflow
# from model import SimpleCNN

# def train():
#     model = SimpleCNN()
#     dummy_input = torch.randn(1,3,224,224)
#     output = model(dummy_input)
#     acc = 0.80  # Dummy accuracy for example
    
#     torch.save(model.state_dict(), "model.pt")
    
#     mlflow.set_experiment("cats_dogs")
#     with mlflow.start_run():
#         mlflow.log_param("lr", 0.001)
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_artifact("model.pt")

# if __name__ == "__main__":
#     train()

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from model import SimpleCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 5
LR = 0.001
BATCH_SIZE = 32

# Create dummy dataset (since we are not using real Kaggle dataset)
X = torch.randn(500, 3, 224, 224)
y = torch.randint(0, 2, (500,))

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

mlflow.set_experiment("cats_dogs")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)

    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Log metrics per epoch
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")

    # ==========================
    # Save Model
    # ==========================
    torch.save(model.state_dict(), "model.pt")
    mlflow.log_artifact("model.pt")

    # ==========================
    # Loss Curve Plot
    # ==========================
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")
    plt.close()

    # ==========================
    # Confusion Matrix
    # ==========================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    print("Training complete. Model and artifacts logged to MLflow.")
