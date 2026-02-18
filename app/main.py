from fastapi import FastAPI, UploadFile
import torch
from src.model import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms
import logging
import time

app = FastAPI()

model = SimpleCNN()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

logging.basicConfig(level=logging.INFO)
request_count = 0


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    global request_count
    request_count += 1
    start_time = time.time()
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    output = model(img)
    prob = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(prob, dim=1).item()

    label_map = {0: "Cat", 1: "Dog"}
    latency = time.time() - start_time
    logging.info(f"Request #{request_count} | Latency: {latency:.4f}s")

    return {
        "predicted_class": label_map[predicted_class],
        "confidence": float(prob[0][predicted_class]),
        "latency": latency
    }

@app.get("/metrics")
def metrics():
    return {
        "total_requests": request_count
    }
