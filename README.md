# 🚀 Adversarial Examples Project

A complete implementation of **adversarial attacks (FGSM, PGD)**, **defenses (adversarial training)**, and **robustness evaluation** using deep learning models.
This project is structured into **backend**, **frontend**, **models**, and **results** for clarity and scalability.

---

# 📁 Folder Structure

```
Adversarial-Examples-Project/
│
├── backend/                # All Python backend code (training, attacks, evaluation)
│     ├── training.py
│     ├── adversarial_backend.py
│     └── utils/            (optional helpers if added later)
│
├── frontend/               # UI components (React/Next.js/Tailwind if completed)
│     ├── Dashboard.tsx
│     ├── globals.css
│     ├── tailwind.config.js
│     └── ...more files i
│
├── models/                 # Trained model weights
│     ├── standard_model.pth
│     ├── robust_model.pth
│     ├── ensemble_model_0.pth
│     ├── ensemble_model_1.pth
│     ├── ensemble_model_2.pth
│
├── results/                # Robustness plots & outputs
│     ├── standard_robustness.png
│     ├── robust_robustness.png
│     └── ensemble_robustness.png
│
└── README.md
```

---

# 🧰 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/hajeeraghazi/Adversarial-Examples-Project.git
cd Adversarial-Examples-Project
```

### 2️⃣ Create & activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r backend/requirements.txt
```

✔ All Python dependencies are now installed.

---

# 📂 Dataset Setup (MNIST / CIFAR-10)

Your backend automatically downloads datasets.

Create a data folder:

```bash
mkdir data
```

Datasets will be downloaded automatically into:

```
data/mnist/
data/cifar10/
```

No manual download required ✔

---

# 🔐 Optional: Create `.env.local`

Inside project root:

```
DATA_DIR=./data
MODEL_DIR=./models
RESULTS_DIR=./results
DEVICE=cuda        # or cpu
SEED=42
```

Not required to run basic scripts, but recommended for paths.

---

# 🚀 How to Run the Project

All backend scripts are inside `/backend`.

Move into backend folder:

```bash
cd backend
```

---

## 🎯 1. Train a Standard Model

```bash
python training.py --mode standard --dataset mnist --epochs 10
```

## 🔒 2. Train a Robust (Adversarially-Trained) Model

```bash
python training.py --mode robust --dataset cifar10 --epochs 10
```

---

## ⚡ 3. Generate Adversarial Examples (FGSM / PGD)

### FGSM:

```bash
python adversarial_backend.py --attack fgsm --eps 0.03 --dataset mnist
```

### PGD:

```bash
python adversarial_backend.py --attack pgd --eps 0.03 --steps 40 --dataset cifar10
```

---

## 🔍 4. Evaluate Clean vs Adversarial Robustness

```bash
python adversarial_backend.py --evaluate
```

Outputs saved to:

```
results/standard_robustness.png
results/robust_robustness.png
results/ensemble_robustness.png
```

---

# 🖥 Optional: Frontend (Dashboard)

Your **frontend** folder contains UI components for visualization.

### If using Next.js or Vite:

```bash
cd frontend
npm install
npm run dev
```

*(Add package.json when frontend is completed)*

---

# 📊 Features

* FGSM & PGD adversarial attack implementation
* Standard model training
* Robust (adversarial) training
* Ensemble model evaluation
* Robustness visualization plots
* Clean folder separation for scalability
* Optional frontend dashboard

---

# 🧪 Ideal For

* ML/AI coursework
* Security & adversarial ML research
* Portfolio & interviews
* Experimenting with attack/defense strategies

---

# 📝 Future Enhancements

* Add more attacks (CW, AutoAttack)
* Add Model Zoo
* Build full frontend dashboard
* Add TensorBoard logging

