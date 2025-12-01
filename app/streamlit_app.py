import streamlit as st
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import json
import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
num_classes = len(label_map)

classifier = nn.Linear(512, num_classes).to(device)
classifier.load_state_dict(
    torch.load("models/facenet_best.pth", map_location=device)
)
classifier.eval()

def predict(image):
    if image is None:
        return "No image", 0.0

    img = image.convert("RGB")
    img_np = np.asarray(img)

    results = face_detector.process(img_np)

    if not results.detections:
        return "Face not detected", 0.0

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    h, w, _ = img_np.shape

    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    face_crop = img_np[y1:y2, x1:x2]

    if face_crop.size == 0:
        return "Invalid face crop", 0.0

    face = cv2.resize(face_crop, (160, 160))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = torch.tensor(face).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face)

    logits = classifier(emb)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    conf = probs[0][pred_idx].item()
    name = label_map.get(pred_idx, "Unknown")

    return name, conf

st.set_page_config(page_title="Face Recognition | FaceNet", layout="wide")

st.title("Face Recognition using FaceNet (Kelompok 6)")
st.write("Upload photo untuk identifikasi mahasiswa")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    name, conf = predict(img)

    with col2:
        if name == "Face not detected":
            st.error("Tidak ada wajah terdeteksi!")
        else:
            st.success(f"Predicted: **{name}**")
            # st.info(f"Confidence: **{conf:.4f}**")
