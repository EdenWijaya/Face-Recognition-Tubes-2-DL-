import streamlit as st
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import json

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}
device = torch.device("cpu")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

num_classes = len(label_map)
classifier = nn.Linear(512, num_classes).to(device)
classifier.load_state_dict(torch.load("models/facenet_best.pth", map_location=device))
classifier.eval()

def predict(image):
    img = image.convert("RGB")

    face = mtcnn(img)

    if face is None:
        return "Face not detected", 0.0

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face)

    logits = classifier(emb)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    conf = probs[0][pred_idx].item()

    name = label_map.get(pred_idx, "Unknown")

    return name, conf

st.set_page_config(page_title="Face Recognition", layout="wide")

st.title("Face Recognition using FaceNet (Kelompok 6)")
st.write("Upload photo untuk identifikasi mahasiswa")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    name, conf = predict(img)

    with col2:
        if name == "Face not detected":
            st.error("Tidak ada wajah terdeteksi!")
        else:
            st.success(f"Predicted: **{name}**")
            st.info(f"Confidence: **{conf:.4f}**")

