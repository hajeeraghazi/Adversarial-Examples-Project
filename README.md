# Adversarial-Examples-Project

This project demonstrates how adversarial inputs can fool deep learning models, and how adversarial training and other defense methods improve robustness. It includes scripts for training models, generating adversarial examples, evaluating robustness, and visualizing results.

---

## 📁 Project Structure

```
root/
│
├── training.py               # Script to train standard / robust model
├── adversarial_backend.py    # Attack generation, evaluation, robustness logic
│
├── standard_model.pth        # Pretrained standard model (optional)
├── robust_model.pth          # Pretrained robust model (optional)
├── ensemble_model_0.pth      # Pretrained ensemble model (optional)
├── ensemble_model_1.pth
├── ensemble_model_2.pth
│
├── standard_robustness.png   # Robustness comparison plots
├── robust_robustness.png
├── ensemble_robustness.png
│
├── requirements.txt          # Python dependencies
└── README.md
```

You may additionally create:

```
data/               # Datasets (created manually)
models/             # Saved models
logs/               # Training & evaluation logs
.env.local          # Environment configuration
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/hajeeraghazi/Adversarial-Examples-Project.git
cd Adversarial-Examples-Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
# or:
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Create `.env.local` File

Create a file named `.env.local` in the **project root** with the following content:

```
DATA_DIR=./data
MODEL_DIR=./models
LOG_DIR=./logs
DEVICE=cuda        # or cpu
SEED=42
```

Explanation:

| Variable    | Purpose                       |
| ----------- | ----------------------------- |
| `DATA_DIR`  | Dataset download or load path |
| `MODEL_DIR` | Saves trained models          |
| `LOG_DIR`   | Stores logs / metrics         |
| `DEVICE`    | `cuda` or `cpu`               |
| `SEED`      | Reproducibility               |

---

## 📁 Create `data/` Folder

Manually create the dataset folder:

```bash
mkdir data
```

If using MNIST / CIFAR-10, PyTorch/TensorFlow will download automatically into:

```
data/mnist/
data/cifar10/
```

Or you can place custom datasets inside `data/`.

---

## 🚀 Running the Project

### ▶️ Train a Standard Model

```bash
python training.py --mode standard
```

### 🔒 Train a Robust (Adversarially-Trained) Model

```bash
python training.py --mode robust
```

### ⚠️ Generate Adversarial Examples (FGSM / PGD)

```bash
python adversarial_backend.py --attack fgsm --eps 0.03
python adversarial_backend.py --attack pgd --eps 0.03 --steps 40
```

### 📊 Evaluate Clean vs Adversarial Robustness

Outputs are saved as PNG plots:

* `standard_robustness.png`
* `robust_robustness.png`
* `ensemble_robustness.png`

---

## 🧪 Features

* Generate adversarial examples (FGSM, PGD).
* Train standard and robust (defended) models.
* Evaluate robustness under different attacks.
* Compare multiple model types (standard, robust, ensemble).
* Visual robustness plots included.

---

## 🚀 Future Improvements

* Add more attack methods (CW, DeepFool, AutoAttack)
* Add GUI/dashboard for visualization
* Add more datasets (CIFAR-100, TinyImageNet)
* Logging with TensorBoard


