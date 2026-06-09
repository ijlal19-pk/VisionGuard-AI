# 👁️ VisionGuard AI

**VisionGuard AI** is an advanced artificial intelligence system designed to screen for Glaucoma using retinal fundus images. Powered by a fine-tuned **EfficientNet-B0** deep neural network, it provides rapid, high-confidence clinical screening augmented with Grad-CAM heatmaps for visual interpretability.

---

## ✨ Features

- **🧠 Deep Learning Diagnosis**: Utilizes transfer learning with a custom EfficientNet-B0 architecture to achieve high sensitivity and specificity in detecting Glaucoma.
- **🗺️ Visual Explainability (Grad-CAM)**: Generates localized heatmaps that highlight the exact regions of the optic disc the neural network focused on, giving clinicians transparent AI insights.
- **⚕️ Clinical Interface**: A sleek, medical-grade web dashboard built with Streamlit and React, designed for ease of use by healthcare professionals.
- **📊 Comprehensive Metrics**: Built-in evaluation tools generating ROC curves, Confusion Matrices, and performance reports.

---

## 🏗️ Architecture & Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core Model** | PyTorch, `torchvision` | Fine-tuned EfficientNet-B0 for binary image classification. |
| **Model Interpretability**| `pytorch-grad-cam` | Provides visual bounding and heatmaps for AI focus areas. |
| **Dashboard Interface** | Streamlit | Rapid UI prototyping and image uploading / previewing. |
| **Full-Stack UI** | React, Vite, FastAPI | Scalable architecture for deployment and API interaction. |

### 📂 Project Structure
```text
VisionGuard AI/
├── app.py               # Streamlit interactive web dashboard
├── backend/             # FastAPI backend for model serving
│   ├── main.py          # API Endpoints
│   └── models.py        # PyTorch model definitions
├── frontend/            # React/Vite web application
├── model/               # Trained .pth model weights (Git Ignored)
├── data/                # ACRIMA Fundus Image Dataset (Git Ignored)
├── notebooks/           # Jupyter notebooks for data exploration
├── report/              # ML PBL Reports and LaTeX files
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
   pip install torch torchvision streamlit pytorch-grad-cam pillow numpy
   ```

### 2. Running the Clinical Dashboard

The easiest way to interact with the model locally is via the Streamlit dashboard:

```bash
streamlit run app.py
```
*The dashboard will automatically open in your default browser.*

### 3. Running the Full Stack App (FastAPI + React)

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

VisionGuard AI was trained on the **ACRIMA dataset**, containing 705 annotated retinal fundus images (396 glaucomatous, 309 normal).

To retrain or evaluate the model:
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
