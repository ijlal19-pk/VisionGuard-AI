# 👁️ VisionGuard AI

**VisionGuard AI** is an advanced artificial intelligence system designed to screen for Glaucoma using retinal fundus images. Powered by a **Knowledge Distillation** architecture, it provides rapid, high-confidence clinical screening augmented with Grad-CAM heatmaps for visual interpretability.

---

## ✨ Features

- **🧠 Knowledge Distillation Architecture**: Utilizes a teacher-student model paradigm. The **ResNet-50** Teacher model provides high-accuracy, slow-processing predictions, while the **EfficientNet-B0** Student model learns from the Teacher to deliver extremely fast and highly efficient real-time inferences.
- **🗺️ Visual Explainability (Grad-CAM)**: Generates localized heatmaps that highlight the exact regions of the optic disc the neural network focused on, giving clinicians transparent AI insights.
- **⚕️ Clinical Interface**: A sleek, medical-grade web dashboard built with React and FastAPI, designed for ease of use by healthcare professionals.
- **📊 Comprehensive Metrics**: Built-in evaluation tools generating ROC curves, Confusion Matrices, and performance reports.

---

## 🏗️ Architecture & Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Teacher Model** | PyTorch (ResNet-50) | High accuracy, robust feature extraction. |
| **Student Model** | PyTorch (EfficientNet-B0)| Fast, lightweight inference mimicking the Teacher. |
| **Model Interpretability**| `pytorch-grad-cam` | Provides visual bounding and heatmaps for AI focus areas. |
| **Backend API** | FastAPI | High-performance backend serving the distillation models. |
| **Frontend UI** | React, Vite | Scalable, responsive web architecture for clinical use. |

### 📂 Project Structure
```text
VisionGuard AI/
├── backend/             # FastAPI backend for model serving
│   ├── main.py          # API Endpoints
│   └── models.py        # PyTorch Teacher & Student model definitions
├── frontend/            # React/Vite web application
├── model/               # Trained .pth model weights (Git Ignored)
├── data/                # ACRIMA Fundus Image Dataset (Git Ignored)
├── notebooks/           # Jupyter notebooks for distillation pipeline
└── results/             # Generated ROC curves and Confusion Matrices
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+**
- **Node.js 16+** (For the React Frontend)

### 1. Model & Backend Setup (Python)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/VisionGuard-AI.git
   cd VisionGuard-AI
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Running the Full Stack App (FastAPI + React)

**Backend API:**
```bash
cd backend
python main.py
```

**React Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 🧠 Model Training & Dataset

VisionGuard AI was trained on the **ACRIMA dataset**, containing 705 annotated retinal fundus images (396 glaucomatous, 309 normal). The training leverages a Distillation Pipeline where the heavy ResNet-50 guides the lightweight EfficientNet-B0.

To retrain or evaluate the models:
```bash
python evaluate.py
```
This generates `metrics.json` and performance plots in the `results/` directory.

---

## ⚠️ Medical Disclaimer

**VisionGuard AI is developed for educational and research purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified ophthalmologist or physician with any questions you may have regarding a medical condition.

---

## 📜 License

This project is licensed under the MIT License.
