import streamlit as st
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import json
import os

# ============================
# LOAD LABEL MAP
# ============================
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

# ============================
# DEVICE SETUP
# (Streamlit Cloud tidak punya GPU → CPU aman)
# ============================
device = torch.device("cpu")

# ============================
# LOAD FACE DETECTOR (MTCNN)
# ============================
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

# ============================
# LOAD FACENET BACKBONE
# ============================
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ============================
# LOAD CLASSIFIER
# ============================
num_classes = len(label_map)
classifier = nn.Linear(512, num_classes).to(device)
classifier.load_state_dict(
    torch.load("models/facenet_best.pth", map_location=device)
)
classifier.eval()


# ==========================================
# PREDICT FUNCTION
# ==========================================
def predict(image):
    if image is None:
        return "No Image", 0.0

    img = image.convert("RGB")

    # ---- TRY FACE DETECTION ---- #
    try:
        face = mtcnn(img)
    except:
        return "Detection Error", 0.0

    if face is None:
        return "Face not detected", 0.0

    face = face.unsqueeze(0).to(device)

    # ---- GET EMBEDDING ---- #
    with torch.no_grad():
        emb = facenet(face)

    # ---- CLASSIFICATION ---- #
    logits = classifier(emb)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    conf = probs[0][pred_idx].item()

    name = label_map.get(pred_idx, "Unknown")

    return name, conf


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Face Recognition | FaceNet", layout="wide")

st.title("🎓 Face Recognition using FaceNet (Kelompok 6)")
st.write("Upload foto wajah, sistem akan memprediksi identitas mahasiswa.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    name, conf = predict(img)

    with col2:
        if name == "Face not detected":
            st.error("🚫 No face detected in the image.")
        elif name == "Detection Error":
            st.error("⚠️ Face detection error occurred.")
        else:
            st.success(f"🎯 Predicted: **{name}**")
            # st.info(f"Confidence: **{conf:.4f}**")
