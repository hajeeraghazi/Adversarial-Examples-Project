# training/train_part1.py
"""
Part 1/2 of the training script.
Contains:
 - imports & config
 - model definition (SimpleCNN)
 - dataset & dataloaders (Windows-safe)
 - FGSM and PGD attack implementations (fixed)
 - helper functions: train_standard, train_adversarial, evaluate, save_model, plot_curves
"""

import os
import math
import random
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import time
import shutil
from pathlib import Path

# -------------------------
# Basic config & reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Make sure script is reasonable on CPU (and uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] device = {device}")

# Limit PyTorch threads to avoid oversubscription on shared machines
torch.set_num_threads(1)

# Directories
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
for d in (MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# Model definition (SimpleCNN)
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # conv blocks
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)   # increased channels for better accuracy
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dims

        # after 3 pools: 32 -> 16 -> 8 -> 4  (so 4x4)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        # weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 16 -> 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 8 -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------
# Dataset + Dataloaders
# -------------------------
# Use CIFAR-10; data will be downloaded if not present
def get_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """
    Returns train_loader, test_loader
    num_workers must be 0 on Windows to avoid spawn issues.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=str(BASE_DIR / "data"), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=str(BASE_DIR / "data"), train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, test_loader

# -------------------------
# Attack implementations
# -------------------------
def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Fast Gradient Sign Method (single-step).
    images: a batch tensor in [0,1]
    returns: adversarial images (clamped to [0,1])
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    # require grad for attack
    images.requires_grad_(True)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # sign of gradient
    sign_grad = images.grad.detach().sign()
    adv = torch.clamp(images + epsilon * sign_grad, 0.0, 1.0)
    return adv.detach()

def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
               epsilon: float, alpha: Optional[float] = None, steps: int = 7) -> torch.Tensor:
    """
    PGD attack (projected gradient descent). Steps kept small (default 7) to balance speed vs strength.
    - alpha: step size (if None uses epsilon/4)
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    if alpha is None:
        alpha = max(0.001, epsilon / 4.0)

    # random start in epsilon-ball
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon).to(device)
    adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    for _ in range(steps):
        # make leaf tensor
        adv = adv.clone().detach().requires_grad_(True)
        outputs = model(adv)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # gradient step
        grad_sign = adv.grad.detach().sign()
        adv = adv + alpha * grad_sign

        # projection onto epsilon-ball around original images
        adv = torch.min(torch.max(adv, images - epsilon), images + epsilon)
        adv = torch.clamp(adv, 0.0, 1.0).detach()

    return adv.detach()

# -------------------------
# Training & evaluation helpers
# -------------------------
def train_standard_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
    """
    Train one epoch on clean images.
    Returns (avg_loss, accuracy%)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train (std)")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.0*correct/total:.2f}%")

    return running_loss / total, 100.0 * correct / total

def train_adversarial_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                            adv_epsilon: float, mix_clean: bool = True) -> Tuple[float, float]:
    """
    One epoch of FGSM-style adversarial training.
    adv_epsilon: epsilon used to craft adversarial examples for training
    mix_clean: if True, trains on clean+adv concatenated batch; else train only on adv
    Returns (avg_loss, accuracy%) on the batch used for training (mixed or adv-only)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train (adv)")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # craft adv examples with current model
        # NOTE: fgsm_attack sets model.eval() internally; that's fine for crafting
        adv_images = fgsm_attack(model, images, labels, adv_epsilon)

        if mix_clean:
            input_batch = torch.cat([images, adv_images], dim=0)
            label_batch = torch.cat([labels, labels], dim=0)
        else:
            input_batch = adv_images
            label_batch = labels

        optimizer.zero_grad()
        outputs = model(input_batch)
        loss = F.cross_entropy(outputs, label_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_batch.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(label_batch).sum().item()
        total += input_batch.size(0)

        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.0*correct/total:.2f}%")

    return running_loss / total, 100.0 * correct / total

def evaluate_model(model: nn.Module, loader: DataLoader, attack: Optional[Callable] = None, eps: float = 0.0,
                   max_batches: Optional[int] = None, pgd_steps: int = 7) -> float:
    """
    Evaluate model accuracy (percentage).
    - attack: None | fgsm_attack | pgd_attack (pass function)
    - eps: epsilon used for attack
    - max_batches: if set, evaluate only on first N batches (useful for fast checks)
    - pgd_steps: steps for PGD if attack is pgd_attack (passed through via lambda)
    """
    model.eval()
    correct = 0
    total = 0

    # We must allow gradients inside the attack function; evaluate() invokes attack which creates gradients,
    # but final forward for scoring can (and should) run under torch.no_grad()
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="eval")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        if attack is None or eps == 0.0:
            # clean evaluation
            with torch.no_grad():
                outputs = model(images)
        else:
            # craft adversarial examples (attack function will call model and backward)
            if attack is pgd_attack:
                # pass steps through a small lambda wrapper
                adv_images = pgd_attack(model, images, labels, epsilon=eps, alpha=eps/4.0, steps=pgd_steps)
            else:
                adv_images = attack(model, images, labels, eps)

            with torch.no_grad():
                outputs = model(adv_images)

        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc

# -------------------------
# Utilities: save model & plot
# -------------------------
def save_model(model: nn.Module, filename: str):
    path = MODELS_DIR / filename
    torch.save(model.state_dict(), str(path))
    print(f"[Save] {path}")

def plot_robustness(epsilons: List[float], fgsm_accs: List[float], pgd_accs: List[float], out_path: Path):
    plt.figure(figsize=(8,5))
    plt.plot(epsilons, fgsm_accs, marker='o', label='FGSM')
    plt.plot(epsilons, pgd_accs, marker='s', label='PGD')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()
    print(f"[Plot] Saved {out_path}")

# Part 2 will include the training orchestration (hyperparams and loops),
# adversarial training run, ensemble training, evaluation & saving final plots & models.
# training/train_part2.py
"""
Part 2/2 - Orchestration
Uses functions defined in Part 1:
 - SimpleCNN, get_dataloaders (get_dataloaders not used here because Part1 created loaders)
 - train_standard_epoch, train_adversarial_epoch, evaluate_model, save_model, plot_robustness
"""
# -------------------------
# Hyperparameters (tunable)
# -------------------------
STANDARD_EPOCHS = 12           # recommended 10-15 for good clean accuracy
ROBUST_EPOCHS = 15             # adversarial training epochs
ENSEMBLE_COUNT = 3
ENSEMBLE_EPOCHS = 5
BATCH_SIZE = 128
ADV_EPSILON = 0.05             # epsilon used for adversarial training and evaluation
PGD_STEPS = 7                  # PGD steps used during evaluation (keep small on CPU)
MAX_EVAL_BATCHES = None        # set to an int (e.g., 5) to speed up evaluation during debugging

# Quick-mode option (set True for fast dev runs, False for full runs)
QUICK_MODE = False
if QUICK_MODE:
    STANDARD_EPOCHS = 3
    ROBUST_EPOCHS = 5
    ENSEMBLE_EPOCHS = 2
    PGD_STEPS = 4
    MAX_EVAL_BATCHES = 5

# Epsilons to evaluate (you can adjust)
EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.1]

# -------------------------
# Prepare dataloaders (if not already created in part1)
# -------------------------
# If Part1 created train_loader/test_loader globally, reuse them.
try:
    _ = train_loader  # type: ignore # noqa: F821
    _ = test_loader # pyright: ignore[reportUndefinedVariable]
except NameError:
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)

# -------------------------
# Training: Standard model
# -------------------------
print("\n[Stage] Training STANDARD model")
model_std = SimpleCNN().to(device)
optim_std = optim.Adam(model_std.parameters(), lr=1e-3)
scheduler_std = optim.lr_scheduler.StepLR(optim_std, step_size=6, gamma=0.5)

for epoch in range(1, STANDARD_EPOCHS + 1):
    start = time.time()
    loss_acc = train_standard_epoch(model_std, train_loader, optim_std)
    elapsed = time.time() - start
    print(f"[Std] Epoch {epoch}/{STANDARD_EPOCHS} done in {elapsed:.1f}s")
    scheduler_std.step()

save_model(model_std, "standard_model.pth")

# -------------------------
# Evaluate standard model robustness
# -------------------------
print("\n[Stage] Evaluating STANDARD model robustness")
fgsm_accs_std = []
pgd_accs_std = []

for eps in EPSILONS:
    print(f"[Eval-STD] epsilon = {eps}")
    fgsm_acc = evaluate_model(model_std, test_loader, attack=fgsm_attack, eps=eps,
                              max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    pgd_acc = evaluate_model(model_std, test_loader, attack=pgd_attack, eps=eps,
                             max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    print(f"  FGSM acc: {fgsm_acc:.2f}%, PGD acc: {pgd_acc:.2f}%")
    fgsm_accs_std.append(fgsm_acc)
    pgd_accs_std.append(pgd_acc)

plot_robustness(EPSILONS, fgsm_accs_std, pgd_accs_std, RESULTS_DIR / "standard_robustness.png")

# -------------------------
# Adversarial training (FGSM-style)
# -------------------------
print("\n[Stage] Training ADVERSARIALLY-ROBUST model (FGSM training)")
model_rob = SimpleCNN().to(device)
optim_rob = optim.Adam(model_rob.parameters(), lr=1e-3)
scheduler_rob = optim.lr_scheduler.StepLR(optim_rob, step_size=7, gamma=0.5)

for epoch in range(1, ROBUST_EPOCHS + 1):
    start = time.time()
    loss_acc = train_adversarial_epoch(model_rob, train_loader, optim_rob, adv_epsilon=ADV_EPSILON, mix_clean=True)
    elapsed = time.time() - start
    print(f"[Robust] Epoch {epoch}/{ROBUST_EPOCHS} done in {elapsed:.1f}s")
    scheduler_rob.step()

save_model(model_rob, "robust_model.pth")

# -------------------------
# Evaluate robust model
# -------------------------
print("\n[Stage] Evaluating ROBUST model robustness")
fgsm_accs_rob = []
pgd_accs_rob = []

for eps in EPSILONS:
    print(f"[Eval-ROB] epsilon = {eps}")
    fgsm_acc = evaluate_model(model_rob, test_loader, attack=fgsm_attack, eps=eps,
                              max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    pgd_acc = evaluate_model(model_rob, test_loader, attack=pgd_attack, eps=eps,
                             max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    print(f"  FGSM acc: {fgsm_acc:.2f}%, PGD acc: {pgd_acc:.2f}%")
    fgsm_accs_rob.append(fgsm_acc)
    pgd_accs_rob.append(pgd_acc)

plot_robustness(EPSILONS, fgsm_accs_rob, pgd_accs_rob, RESULTS_DIR / "robust_robustness.png")

# -------------------------
# Ensemble training (optional)
# -------------------------
print("\n[Stage] Training ENSEMBLE models")
ensemble_models = []
for i in range(ENSEMBLE_COUNT):
    print(f"[Ensemble] Training model {i+1}/{ENSEMBLE_COUNT}")
    m = SimpleCNN().to(device)
    optim_e = optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(1, ENSEMBLE_EPOCHS + 1):
        train_adversarial_epoch(m, train_loader, optim_e, adv_epsilon=ADV_EPSILON, mix_clean=True)
    save_model(m, f"ensemble_model_{i}.pth")
    ensemble_models.append(m)

# -------------------------
# Evaluate ensemble by averaging logits
# -------------------------
def evaluate_ensemble(models: List[nn.Module], loader: DataLoader, attack=None, eps=0.0, max_batches=None, pgd_steps=7):
    for mod in models:
        mod.eval()
    total, correct = 0, 0
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="eval ensemble")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)

        if attack is not None and eps > 0:
            if attack is pgd_attack:
                adv = pgd_attack(models[0], images, labels, epsilon=eps, alpha=eps/4.0, steps=pgd_steps)
            else:
                adv = attack(models[0], images, labels, eps)
            with torch.no_grad():
                # average logits
                logits_sum = None
                for m in models:
                    logits = m(adv)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                logits_avg = logits_sum / len(models)
                preds = logits_avg.argmax(1)
        else:
            with torch.no_grad():
                logits_sum = None
                for m in models:
                    logits = m(images)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                logits_avg = logits_sum / len(models)
                preds = logits_avg.argmax(1)

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0

print("\n[Stage] Evaluating ENSEMBLE robustness (on same EPS list)")
fgsm_accs_ens = []
pgd_accs_ens = []
for eps in EPSILONS:
    print(f"[Eval-ENS] epsilon = {eps}")
    fa = evaluate_ensemble(ensemble_models, test_loader, attack=fgsm_attack, eps=eps, max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    pa = evaluate_ensemble(ensemble_models, test_loader, attack=pgd_attack, eps=eps, max_batches=MAX_EVAL_BATCHES, pgd_steps=PGD_STEPS)
    print(f"  Ensemble FGSM acc: {fa:.2f}%, PGD acc: {pa:.2f}%")
    fgsm_accs_ens.append(fa)
    pgd_accs_ens.append(pa)

plot_robustness(EPSILONS, fgsm_accs_ens, pgd_accs_ens, RESULTS_DIR / "ensemble_robustness.png")

# -------------------------
# Copy models into backend (if exists)
# -------------------------
BACKEND_MODELS_DIR = Path.cwd().parent / "backend" / "models"
if BACKEND_MODELS_DIR.exists():
    print(f"\n[Deploy] Copying models to backend: {BACKEND_MODELS_DIR}")
    for fname in ["standard_model.pth", "robust_model.pth"] + [f"ensemble_model_{i}.pth" for i in range(ENSEMBLE_COUNT)]:
        src = MODELS_DIR / fname
        dst = BACKEND_MODELS_DIR / fname
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"  Copied {src} -> {dst}")
        else:
            print(f"  Missing {src}, skipping")
else:
    print("\n[Deploy] backend/models not found — skipping copy. To deploy, create backend/models and re-run or copy files manually.")

print("\nALL DONE. Models in /models, plots in /results.")
print("To run backend with trained models: copy the .pth files to backend/models and restart uvicorn.")