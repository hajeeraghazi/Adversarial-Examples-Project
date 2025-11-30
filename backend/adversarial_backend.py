"""
FastAPI Backend for Adversarial Dashboard
Updated to match EXACT training model architecture.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CORS + APP
# ============================================================

app = FastAPI(title="Adversarial Examples API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENUMS FOR FRONTEND
# ============================================================

class AttackType(str, Enum):
    fgsm = "fgsm"
    pgd = "pgd"

class DefenseType(str, Enum):
    none = "none"
    adversarial = "adversarial"
    input_transform = "input_transform"
    ensemble = "ensemble"


# ============================================================
# TRAINING MODEL (EXACT COPY OF YOUR TRAINING MODEL)
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================
# ATTACKS
# ============================================================

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().requires_grad_(True)
    labels = labels

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv = images + epsilon * images.grad.sign()
    return torch.clamp(adv, 0, 1).detach()


def pgd_attack(model, images, labels, epsilon, alpha=0.01, steps=7):
    ori = images.clone().detach()
    adv = ori + torch.empty_like(ori).uniform_(-epsilon, epsilon)
    adv = adv.clamp(0, 1)

    for _ in range(steps):
        adv = adv.clone().detach().requires_grad_(True)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv = adv + alpha * adv.grad.sign()
        adv = torch.max(torch.min(adv, ori + epsilon), ori - epsilon)
        adv = adv.clamp(0, 1).detach()

    return adv


def input_transform(images):
    reduced = torch.round(images * 10) / 10
    noise = torch.randn_like(images) * 0.01
    return (reduced + noise).clamp(0, 1)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ExperimentRequest(BaseModel):
    attack_type: AttackType
    epsilon: float
    defense_type: DefenseType
    num_samples: int = 200

class ExperimentResponse(BaseModel):
    clean_accuracy: float
    attack_success_rate: float
    robust_accuracy: float
    perturbation_magnitude: float
    samples_evaluated: int


# ============================================================
# MODEL LOADER / MANAGER
# ============================================================

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False

    def init(self):
        if self.initialized:
            return

        print("⏳ Initializing CIFAR-10 + models...")

        transform = transforms.Compose([transforms.ToTensor()])

        test_set = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        self.test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Load all models
        self.standard_model = self._load_model("./models/standard_model.pth")
        self.robust_model = self._load_model("./models/robust_model.pth")

        self.ensemble_models = []
        for i in range(3):
            path = f"./models/ensemble_model_{i}.pth"
            if os.path.exists(path):
                self.ensemble_models.append(self._load_model(path))

        print("✅ Backend model initialization complete.")
        self.initialized = True

    def _load_model(self, path):
        model = SimpleCNN().to(self.device)
        if os.path.exists(path):
            print(f"Loading model: {path}")
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print(f"⚠ Model file missing: {path}")
        model.eval()
        return model


manager = ModelManager()


@app.on_event("startup")
async def startup():
    manager.init()


# ============================================================
# MAIN EXPERIMENT ENDPOINT
# ============================================================

@app.post("/api/run_experiment", response_model=ExperimentResponse)
async def run_experiment(req: ExperimentRequest):

    model = manager.standard_model  # default

    if req.defense_type == DefenseType.adversarial:
        model = manager.robust_model

    device = manager.device
    model.to(device).eval()

    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in manager.test_loader:
        if total >= req.num_samples:
            break

        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_pred = model(images)
            clean_correct += (clean_pred.argmax(1) == labels).sum().item()

        # Apply attack
        if req.attack_type == AttackType.fgsm:
            adv = fgsm_attack(model, images, labels, req.epsilon)
        else:
            adv = pgd_attack(model, images, labels, req.epsilon)

        # Defense: input transform
        if req.defense_type == DefenseType.input_transform:
            adv = input_transform(adv)

        # Ensemble defense
        if req.defense_type == DefenseType.ensemble and len(manager.ensemble_models) > 0:
            logits = None
            for m in manager.ensemble_models:
                out = m(adv)
                logits = out if logits is None else logits + out
            adv_pred = logits / len(manager.ensemble_models)
        else:
            with torch.no_grad():
                adv_pred = model(adv)

        adv_correct += (adv_pred.argmax(1) == labels).sum().item()
        total += labels.size(0)

    clean_acc = clean_correct * 100 / total
    robust_acc = adv_correct * 100 / total
    attack_success = 100 - robust_acc

    return ExperimentResponse(
        clean_accuracy=round(clean_acc, 2),
        attack_success_rate=round(attack_success, 2),
        robust_accuracy=round(robust_acc, 2),
        perturbation_magnitude=req.epsilon,
        samples_evaluated=total
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "initialized": manager.initialized}


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("adversarial_backend:app", host="0.0.0.0", port=8000, reload=True)
