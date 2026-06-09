# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import json
import time

# Page Config
st.set_page_config(
    page_title="VisionGuard AI - Glaucoma Screening",
    page_icon="eye",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BASE_DIR = r"c:\Users\ijlal\Desktop\GLUCOMA DETECTOR"
MODEL_PATH = os.path.join(BASE_DIR, "model", "glaucoma_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# Custom CSS - Using only ASCII-safe characters
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #0e2439, #1a1a40);
        border: 1px solid #1e90ff;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title {
        color: #22d3ee;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.4rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-size: 2rem;
        font-weight: 700;
        color: #22d3ee;
    }
    .stat-lbl {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 4px;
    }
    .diag-glaucoma {
        background: #2d1215;
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
    }
    .diag-glaucoma h2 { color: #ef4444; font-weight: 700; margin: 0; }
    .diag-glaucoma p  { color: #fca5a5; margin-top: 0.4rem; }
    .diag-normal {
        background: #0d291a;
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
    }
    .diag-normal h2 { color: #22c55e; font-weight: 700; margin: 0; }
    .diag-normal p  { color: #86efac; margin-top: 0.4rem; }
    .sec-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 600;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1e90ff;
        margin: 1.8rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Load Model (cached)
@st.cache_resource
def load_model():
    m = models.efficientnet_b0()
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()
    return m


def predict(image, mdl):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = mdl(tensor)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), tensor


def get_heatmap(tensor, orig_img, mdl):
    cam = GradCAM(model=mdl, target_layers=[mdl.features[-1]])
    gc = cam(input_tensor=tensor)[0, :]
    arr = np.array(orig_img.resize((224, 224))).astype(np.float32) / 255.0
    return show_cam_on_image(arr, gc, use_rgb=True)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# =================== UI ===================

# Hero Section
st.markdown("""
<div class="hero-box">
    <div class="hero-title">VisionGuard AI</div>
    <div class="hero-sub">Advanced AI-Powered Glaucoma Screening from Retinal Fundus Images</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About VisionGuard")
    st.write(
        "VisionGuard uses a fine-tuned **EfficientNet-B0** deep neural network "
        "trained on the **ACRIMA dataset** (705 fundus images) to detect signs of glaucoma."
    )
    st.divider()

    st.subheader("Model Details")
    st.markdown("""
    | Property | Value |
    |:---|:---|
    | Architecture | EfficientNet-B0 |
    | Parameters | 5.3 M |
    | Input Size | 224 x 224 |
    | Training | Transfer Learning |
    | Dataset | ACRIMA |
    """)

    metrics = load_metrics()
    if metrics:
        st.divider()
        st.subheader("Test Performance")
        st.metric("Accuracy", f"{metrics['accuracy']}%")
        st.metric("Sensitivity", f"{metrics['sensitivity']}%")
        st.metric("Specificity", f"{metrics['specificity']}%")
        st.metric("AUC-ROC", f"{metrics['auc_roc']}")

# Metric Cards (main area)
metrics = load_metrics()
if metrics:
    st.markdown('<div class="sec-title">Model Performance Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{metrics['accuracy']}%", "Accuracy"),
        (c2, f"{metrics['sensitivity']}%", "Sensitivity"),
        (c3, f"{metrics['specificity']}%", "Specificity"),
        (c4, f"{metrics['auc_roc']}", "AUC-ROC"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    # Evaluation charts
    plots = []
    for title, fname in [("Confusion Matrix", "confusion_matrix.png"),
                         ("ROC Curve", "roc_curve.png"),
                         ("Training Curves", "training_curves.png")]:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            plots.append((title, p))

    if plots:
        with st.expander("View Evaluation Charts", expanded=False):
            cols = st.columns(len(plots))
            for i, (t, p) in enumerate(plots):
                cols[i].image(p, caption=t, use_container_width=True)

# Upload and Predict
st.markdown('<div class="sec-title">Upload Fundus Image for Screening</div>', unsafe_allow_html=True)
st.info("You can grab test images from the data/test/glaucoma/ or data/test/normal/ folders in your project.")

uploaded = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    model = load_model()
    image = Image.open(uploaded).convert("RGB")

    bar = st.progress(0, text="Initializing...")
    for i in range(100):
        time.sleep(0.008)
        if i < 30:
            bar.progress(i + 1, text="Loading retinal image...")
        elif i < 60:
            bar.progress(i + 1, text="Analyzing optic disc structures...")
        elif i < 85:
            bar.progress(i + 1, text="Running neural network inference...")
        else:
            bar.progress(i + 1, text="Generating Grad-CAM heatmap...")

    pred_idx, confidence, tensor = predict(image, model)
    heatmap = get_heatmap(tensor, image, model)
    bar.empty()

    # Images side by side
    st.markdown('<div class="sec-title">Analysis Results</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    left.image(image, caption="Original Fundus Image", use_container_width=True)
    right.image(heatmap, caption="Grad-CAM - AI Focus Regions", use_container_width=True)

    # Diagnosis
    classes = ["Glaucoma Detected", "Normal / Healthy"]
    is_glaucoma = pred_idx == 0
    st.divider()

    if is_glaucoma:
        st.markdown(
            f'<div class="diag-glaucoma"><h2>WARNING: {classes[pred_idx]}</h2>'
            f'<p>Confidence: <strong>{confidence*100:.1f}%</strong></p></div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "**Clinical Advisory:** The AI model has detected features consistent with glaucoma. "
            "Please consult a qualified ophthalmologist for a comprehensive clinical examination."
        )
    else:
        st.markdown(
            f'<div class="diag-normal"><h2>CLEAR: {classes[pred_idx]}</h2>'
            f'<p>Confidence: <strong>{confidence*100:.1f}%</strong></p></div>',
            unsafe_allow_html=True,
        )
        st.success(
            "**Result:** The retinal image appears healthy based on AI analysis. "
            "Regular eye check-ups are still recommended."
        )

    # Confidence breakdown
    st.markdown('<div class="sec-title">Confidence Breakdown</div>', unsafe_allow_html=True)
    g_prob = confidence * 100 if is_glaucoma else (1 - confidence) * 100
    n_prob = confidence * 100 if not is_glaucoma else (1 - confidence) * 100
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Glaucoma Probability**")
        st.progress(g_prob / 100)
        st.code(f"{g_prob:.1f}%")
    with b2:
        st.markdown("**Normal Probability**")
        st.progress(n_prob / 100)
        st.code(f"{n_prob:.1f}%")

# Footer
st.divider()
st.caption("VisionGuard AI - For Educational Purposes Only. Not a substitute for professional medical diagnosis.")
