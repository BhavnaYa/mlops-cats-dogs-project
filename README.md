# MLOPS Assignment 2 : Group 111
SHAIK MAHAMMED ASIF : 2024aa05500

BHAVNA YADAV        : 2023ac05950

K JAGADEESH KUMAR   : 2024aa05124

M SWATI RANI        : 2024aa05305

V N SANJAY          : 2024aa05123

---

# 🐶🐱 Cats vs Dogs – End-to-End MLOps Pipeline

This project implements a complete **end-to-end MLOps pipeline** for a binary image classification task (Cats vs Dogs) designed for a pet adoption platform.

It demonstrates model development, experiment tracking, packaging, containerization, CI/CD automation, deployment validation, and post-deployment monitoring using open-source tools.

---

# 📌 Project Overview

**Use Case:**
Binary classification of cat and dog images.

**Objective:**
Design and implement a reproducible and automated MLOps pipeline using industry-standard open-source tools.

---

# 🏗 End-to-End Architecture

Data → Model Training → MLflow Tracking → Model Artifact
↓
FastAPI Inference Service
↓
Docker Containerization
↓
GitHub Actions CI/CD
↓
Deployment + Smoke Testing
↓
Post-Deployment Monitoring

---

# 📂 Project Structure

```
mlops-cats-dogs-project/
│
├── app/                      # FastAPI inference service
├── src/                      # Model + training code
├── tests/                    # Unit tests
├── test_images/              # Images for post-deployment evaluation
│
├── train.py                  # Model training + MLflow logging
├── evaluate_post_deploy.py   # Production monitoring script
├── smoke_check.py            # Smoke test script
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── .github/workflows/ci.yml  # CI/CD pipeline
├── data.dvc                  # DVC dataset tracking
└── README.md
```

---

# 🧠 M1 – Model Development & Experiment Tracking

## ✅ Model

* Baseline CNN implemented using PyTorch
* Input images resized to 224x224 RGB
* Model saved as `model.pt`

## ✅ Experiment Tracking (MLflow)

Logged:

* Hyperparameters (epochs, learning rate, batch size)
* Training loss
* Training accuracy
* Loss curve artifact
* Confusion matrix artifact
* Trained model artifact

### Run Training

```
python src/train.py
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

# 📦 M2 – Model Packaging & Containerization

## ✅ FastAPI Inference Service

Endpoints:

| Endpoint   | Description                                    |
| ---------- | ---------------------------------------------- |
| `/health`  | Health check                                   |
| `/predict` | Returns predicted class + confidence + latency |
| `/metrics` | Returns total request count                    |

### Run API Locally

```
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## ✅ Docker

Build Image:

```
docker build -t cats-dogs .
```

Run Container:

```
docker run -p 8000:8000 cats-dogs
```

---

# 🔄 M3 – Continuous Integration (CI)

Implemented using **GitHub Actions**.

Pipeline automatically:

* Installs dependencies
* Runs unit tests (pytest)
* Builds Docker image
* Logs into DockerHub
* Pushes image to registry
* Starts container
* Executes smoke test

Triggered on every push to `main`.

---

# 🚀 M4 – Continuous Deployment (CD)

* Docker image automatically pushed to DockerHub
* Container deployed during CI
* Smoke test validates API availability
* Deployment script (`deploy.sh`) included

---

# 📊 M5 – Monitoring & Post-Deployment Tracking

## ✅ Request Logging

Each API call logs:

* Request number
* Latency

Example:

```
Request #1 | Latency: 0.0404s
```

---

## ✅ Monitoring Endpoint

```
GET /metrics
```

Returns:

```json
{
  "total_requests": 5
}
```

---

## ✅ Post-Deployment Performance Tracking

Script:

```
python evaluate_post_deploy.py
```

This:

* Sends test images to deployed API
* Compares predicted vs true labels
* Calculates production accuracy
* Logs metric to MLflow

Example output:

```
Post-Deployment Accuracy: 0.25
```

---

# 📦 Dataset Versioning (DVC)

Dataset tracked using DVC:

```
dvc init
dvc add data/
git add data.dvc
```

Ensures reproducible dataset management separate from source code.

---

# 🛠 Technologies Used

* Python
* PyTorch
* FastAPI
* MLflow
* DVC
* Docker
* GitHub Actions
* Pytest

---

# ▶️ How To Run End-to-End

### 1️⃣ Train Model

```
python src/train.py
```

### 2️⃣ Start API

```
uvicorn app.main:app --reload
```

### 3️⃣ Test Prediction

```
curl -X POST -F "file=@dog.jpg" http://127.0.0.1:8000/predict
```

### 4️⃣ Monitor Requests

Open:

```
http://127.0.0.1:8000/metrics
```

### 5️⃣ Run Post-Deployment Evaluation

```
python evaluate_post_deploy.py
```

---

# 🎓 Assignment Coverage

| Module                    | Status |
| ------------------------- | ------ |
| M1 – Model & Tracking     | ✅      |
| M2 – Packaging & Docker   | ✅      |
| M3 – CI Pipeline          | ✅      |
| M4 – CD Deployment        | ✅      |
| M5 – Monitoring & Logging | ✅      |

---

# 🏆 Conclusion

This project demonstrates a complete, automated, and reproducible MLOps pipeline integrating:

* Model training
* Experiment tracking
* API development
* Docker containerization
* CI/CD automation
* Deployment validation
* Production monitoring

It reflects real-world MLOps practices used in industry systems.

---

## 👩‍💻 Author

Bhavna Ya