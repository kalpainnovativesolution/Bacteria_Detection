import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
import os
import gdown

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI-powered computer vision system for surface hygiene monitoring",
    layout="wide"
)

# =========================================================
# HELPER: LOAD LOGO
# =========================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("prompt_logo_transparent.png")

# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #2b2a6a;
        padding-top: 20px;
        font-family: 'Poppins', sans-serif;
    }

    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 25px;
    }

    .logo {
        height: 150px;
    }

    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }

    .title-text {
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
    }

    .subtitle-text {
        font-size: 18px;
        color: #cbd5e1;
        margin-top: 6px;
    }

    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: 2px dashed rgba(255,255,255,0.4);
        border-radius: 14px;
        padding: 28px;
    }

    div[data-testid="stFileUploader"] label {
        display: none;
    }

    div[data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
        font-weight: 500;
    }

    div[data-testid="stFileUploaderFileSize"] {
        color: #e5e7eb !important;
    }

    .upload-subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 16px;
    }

    .result-card {
        max-width: 620px;
        margin: 40px auto 0 auto;
        padding: 28px 32px;
        background: linear-gradient(
            180deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.02)
        );
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        text-align: center;
    }

    .result-header {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
    }

    .result-divider {
        height: 1px;
        background: rgba(255,255,255,0.25);
        margin: 18px 0;
    }

    .result-metric {
        font-size: 18px;
        font-weight: 500;
        color: #ffffff;
        margin: 10px 0;
    }

    .result-clean {
        font-size: 26px;
        font-weight: 800;
        color: #7dd3fc;
    }

    .info-text {
        color: #ffffff;
        text-align: center;
        font-size: 16px;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    f"""
    <div class="top-bar">
        <img src="data:image/png;base64,{logo_base64}" class="logo">
    </div>

    <div class="title-container">
        <div class="title-text">
            AI-powered computer vision system for surface hygiene monitoring
        </div>
        <div class="subtitle-text">
            Upload exactly 2 images for detection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD YOLO MODEL FROM GOOGLE DRIVE (FOLDER)
# =========================================================
GDRIVE_FOLDER_ID = "138FKWerRv6A5jgPh4LtEuaSi4KdrnaWJ"
MODEL_NAME = "Yolov11_BacteriaDetection.pt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def load_yolo_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=MODEL_DIR,
                quiet=False
            )

    return YOLO(MODEL_PATH)

yolo_model = load_yolo_model()

# =========================================================
# YOLO INFERENCE FUNCTION
# =========================================================
def run_yolo_and_get_counts(img_pil, model, conf_threshold=0.25):
    img_np = np.array(img_pil)
    results = model(img_np, conf=conf_threshold, iou=0.5)

    counts = {}
    if results and results[0].boxes is not None:
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        for cls_id in classes:
            label = results[0].names[int(cls_id)].lower().strip()
            counts[label] = counts.get(label, 0) + 1
    return counts

# =========================================================
# FILE UPLOADER
# =========================================================
st.markdown(
    "<div class='upload-subtitle'>Drag and drop exactly 2 images or click to browse</div>",
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    label="Surface Image Upload",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="surface_image_uploader",
    label_visibility="collapsed"
)

# =========================================================
# RUN DETECTION
# =========================================================
if uploaded_files and len(uploaded_files) == 2:

    img1 = Image.open(uploaded_files[0]).convert("RGB")
    img2 = Image.open(uploaded_files[1]).convert("RGB")

    counts1 = run_yolo_and_get_counts(img1, yolo_model)
    counts2 = run_yolo_and_get_counts(img2, yolo_model)

    total_counts = {}
    for d in (counts1, counts2):
        for k, v in d.items():
            total_counts[k] = total_counts.get(k, 0) + v

    bacteria = total_counts.get("bacteria", 0)
    debries = total_counts.get("debries", 0)
    milk = total_counts.get("milk_residues", 0)

    if sum(total_counts.values()) == 0:
        card_html = """
        <div class="result-card">
            <div class="result-header">Detection Result</div>
            <div class="result-divider"></div>
            <div class="result-clean">Surface is Clean</div>
        </div>
        """
    elif bacteria > 0:
        bacteria_ml = bacteria * 1000
        cfu = bacteria_ml * 0.09
        card_html = f"""
        <div class="result-card">
            <div class="result-header">Bacterial Contamination Detected</div>
            <div class="result-divider"></div>
            <div class="result-metric">Bacteria/ml: <b>{bacteria_ml}</b></div>
            <div class="result-metric">CFU/ml: <b>{cfu:.2f}</b></div>
        </div>
        """
    else:
        metrics = ""
        if milk > 0:
            metrics += f"<div class='result-metric'>Milk Residues/ml: <b>{milk * 1000}</b></div>"
        if debries > 0:
            metrics += f"<div class='result-metric'>Debries/ml: <b>{debries * 1000}</b></div>"

        card_html = f"""
        <div class="result-card">
            <div class="result-header">Surface Residue Detected</div>
            <div class="result-divider"></div>
            {metrics}
        </div>
        """

    st.markdown(card_html, unsafe_allow_html=True)

elif uploaded_files:
    st.warning("Please upload exactly 2 images.")

else:
    st.markdown(
        "<div class='info-text'>Please upload exactly 2 images to run the detection.</div>",
        unsafe_allow_html=True
    )
