# Adversarial-Examples-Project

This project explores adversarial attacks and defenses on deep learning image classifiers.  
It allows training of standard and robust models, generating adversarial examples, and comparing robustness/performance under attack.

---

## 📂 Repository Structure

```

root/
│
├── training.py               # Script to train standard / robust model
├── adversarial_backend.py    # Attack generation, evaluation, robustness logic
│
├── standard_model.pth        # (optional) Pretrained standard model
├── robust_model.pth          # (optional) Pretrained robust model
├── ensemble_model_0.pth      # (optional) Part of ensemble model
├── ensemble_model_1.pth
├── ensemble_model_2.pth
│
├── standard_robustness.png   # Robustness result plots for standard model
├── robust_robustness.png     # Robustness result plots for robust model
├── ensemble_robustness.png   # Robustness result plots for ensemble model
│
├── requirements.txt          # Python dependencies
│
├── frontend files            # (TypeScript, CSS) — possibly for UI / dashboard
│   ├── Dashboard.tsx
│   ├── tailwind.config.js
│   └── globals.css
└── README.md                 # <- This file

````

> ⚠️ Note: There is currently **no `data/` folder**, and **no `.env.local` file** in the repository.

---

## ✅ Prerequisites

- Python 3.x  
- (Optional but recommended) A virtual environment  
- Required packages as per `requirements.txt`

---

## 🔧 Setup & Installation

```bash
git clone https://github.com/hajeeraghazi/Adversarial-Examples-Project.git
cd Adversarial-Examples-Project

# (Recommended) Create a virtual environment:
python -m venv venv
source venv/bin/activate     # Linux / macOS
# or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
````

---

## 🔐 Configuration (Optional but Recommended)

Create a file named `.env.local` at the project root (not included in repo) with configuration variables.
Example:

```
DATA_DIR=./data
MODEL_DIR=./models
LOG_DIR=./logs
DEVICE=cuda   # or cpu
SEED=42
```

* `DATA_DIR` → where dataset (e.g. MNIST, CIFAR-10) will be downloaded or stored
* `MODEL_DIR` → where trained model checkpoints will be saved/read
* `LOG_DIR` → for logs, metrics, or other outputs
* `DEVICE` → `cuda` or `cpu`, depending on GPU availability
* `SEED` → for reproducible results

If you do not use `.env.local`, ensure that defaults in your code paths correspond to actual folders or modify the code accordingly.

---

## 📁 Data Folder (Manually Create If Needed)

Since there is currently **no `data/` folder** in the repo, if your code assumes dataset files locally, you should create:

```
data/
  ├── mnist/      # or whichever dataset you use
  └── cifar10/
```

Alternatively, if your scripts are designed to auto-download datasets, ensure internet connection is available when running for the first time.

Example to create folder manually:

```bash
mkdir -p data/mnist data/cifar10
```

---

## 🚀 Usage / Workflow

### 1. Train a Model (Standard or Robust)

```bash
python training.py --mode standard   # train a normal model
python training.py --mode robust     # train an adversarially-trained model
```

Replace arguments (dataset, epochs, etc.) according to your script’s parameters.

### 2. Generate Adversarial Examples & Evaluate Robustness

```bash
python adversarial_backend.py --attack fgsm   --eps 0.03 --dataset MNIST
python adversarial_backend.py --attack pgd    --eps 0.03 --steps 40 --dataset CIFAR10
```

This should create adversarial examples, run inference under attack with your model(s), and output robustness metrics/plots (like `*_robustness.png`).

### 3. (Optional) Explore Frontend / Dashboard

There appears to be a UI component (TypeScript + CSS) — if you intend to enable a dashboard:

* Ensure you have Node.js / npm installed
* Add appropriate config (e.g. `package.json`)
* Install dependencies and run the frontend (instructions need to be added)

---

## 📊 What This Project Demonstrates

* Training of clean (standard) and robust (defended) models
* Generation of adversarial examples using common attacks (e.g. FGSM, PGD)
* Evaluation and comparison of model robustness under adversarial attacks
* (Optional) Visualization or UI for comparing results

---

## 🛠 Suggestions / Next Steps (To Make Project More Complete)

* Add a `.env.local` or configuration management to define paths
* Add (or enable) a `data/` folder or dataset download logic
* Add argument-parsing and README instructions for all command-line options
* If using the frontend, add `package.json`, build scripts, and instructions for launching the UI
* Add more documentation about what each script does, expected inputs/outputs

