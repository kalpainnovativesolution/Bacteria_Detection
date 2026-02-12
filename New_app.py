import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
import cv2
import os
import gdown

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SOMAEYE: Vision-Based Surface Hygiene & Bacteria Load Analysis",
    layout="wide"
)

# =========================================================
# SESSION STATE INIT
# =========================================================
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

# =========================================================
# HELPER: LOAD LOGO
# =========================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("SOMAEYE-Bacteria.jpeg")

# =========================================================
# CUSTOM CSS (ONLY FINAL RESULT COLORS + FONT SIZE)
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(90deg, #d8f1ff 0%, #eef8ff 100%);
}

.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
    border-left: 6px solid;
    margin: 12px 0;
}

.card.clean { border-color: #22c55e; }
.card.critical { border-color: #ef4444; }
.card.parrot-green { border-color: #00FF00; }
.card.info { border-color: #3b82f6; }

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #475569;
}

.card-value {
    font-size: 36px;
    font-weight: 900;
    color: #0f172a;
}
            
/* ===== CHANGE BUTTON TEXT ONLY ===== */
[data-testid="stFileUploader"] button {
    font-size: 0px;
}
[data-testid="stFileUploader"] button::after {
    content: "Capture Image";
    font-size: 16px;
    font-weight: 600;
}

/* FINAL RESULT ONLY */
.final-card {
    border-radius: 18px;
    padding: 32px;
    margin: 22px 0;
    color: white;
    text-align: center;
    font-weight: 900;
    font-size: 34px;
}

.final-clean { background-color: #22c55e; }
.final-critical { background-color: #ef4444; }
.final-caution { background-color: #f59e0b; } /* ORANGE-YELLOW */
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:20px;">
        <img src="data:image/png;base64,{logo_base64}" style="width:360px;">
        <h1>AI-Powered Surface Hygiene Verification For CIP in Dairy Processing</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CLASS COLORS
# =========================================================
CLASS_COLORS = {
    "bacteria": {"bgr": (0, 0, 255)},
    "milk_residues": {"bgr": (0, 255, 0)},
    "debries": {"bgr": (255, 0, 0)}
}

# =========================================================
# LOAD YOLO MODEL FROM GOOGLE DRIVE (SINGLE FILE)
# =========================================================
GDRIVE_FILE_ID = "1wYIHhpl_aCHKui3yMo-7I8kqucmUtED2"
MODEL_NAME = "Yolov11_BacteriaDetection.pt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def load_yolo_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model from Google Drive..."):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    return YOLO(MODEL_PATH)

if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = load_yolo_model()

# =========================================================
# YOLO INFERENCE
# =========================================================
def run_yolo(img_pil, model):
    img = np.array(img_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model(img, conf=0.25, iou=0.5)
    counts = {}

    if results and results[0].boxes is not None:
        for box, cls_id in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy().astype(int)
        ):
            label = results[0].names[int(cls_id)].lower().strip()
            counts[label] = counts.get(label, 0) + 1

            color = CLASS_COLORS.get(label, {}).get("bgr", (0, 255, 0))
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    return counts, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Capture 1 image for Surface Hygiene Verification",
    type=["jpg", "jpeg", "png"],
    key=f"image_uploader_{st.session_state.uploader_version}"
)

# =========================================================
# MAIN EXECUTION
# =========================================================
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    counts, ann_img = run_yolo(img, st.session_state.yolo_model)

    bacteria = counts.get("bacteria", 0)
    milk = counts.get("milk_residues", 0)
    debries = counts.get("debries", 0)

    # IMAGE DISPLAY
    st.image(ann_img, caption="Detection Output", use_container_width=True)

    # COUNTS
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"<div class='card critical'><div class='card-title'>Bacteria Count</div><div class='card-value'>{bacteria}</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='card parrot-green'><div class='card-title'>Milk Residues Count</div><div class='card-value'>{milk}</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='card info'><div class='card-title'>Debries Count</div><div class='card-value'>{debries}</div></div>", unsafe_allow_html=True)

    # BACTERIA / ML & CFU
    if bacteria > 0:
        bacteria_ml = int(bacteria * 1000)
        cfu = int(np.round(bacteria_ml / 3))

        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='card critical'><div class='card-title'>Bacteria / ml</div><div class='card-value'>{bacteria_ml}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card critical'><div class='card-title'>CFU / ml</div><div class='card-value'>{cfu}</div></div>", unsafe_allow_html=True)

    # =====================================================
    # FINAL RESULT (WITH CAUTION CONDITION)
    if bacteria > 15 or milk > 10 or debries > 10:
        st.markdown("""
        <div class="final-card final-critical">
            ✖ Surface Is Not Clean
        </div>
        """, unsafe_allow_html=True)

    elif (5 <= bacteria <= 15) or (5 <= milk <= 10) or (5 <= debries <= 10):
        st.markdown("""
        <div class="final-card final-caution">
            ⚠️ Caution
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="final-card final-clean">
            ✅ Surface Is Clean
        </div>
        """, unsafe_allow_html=True)

    # NEXT SAMPLE
    st.markdown("---")
    if st.button("🔄 Test Next Sample"):
        st.session_state.uploader_version += 1
        st.rerun()

else:
    st.info("Please capture 1 image to check surface hygiene")
