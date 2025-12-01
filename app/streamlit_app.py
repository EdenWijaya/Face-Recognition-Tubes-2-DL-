import streamlit as st
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import json
import os

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

# Convert string keys to int
label_map = {int(k): v for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

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
    img = np.asarray(img).astype(np.uint8)

    if len(img.shape) != 3 or img.shape[2] != 3:
        return "Invalid image format", 0.0
    
    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        return "Face not detected", 0.0

    # Ambil wajah pertama
    x1, y1, x2, y2 = boxes[0].astype(int)

    # Proteksi index out of bounds
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return "Invalid face crop", 0.0

    # Convert crop menjadi tensor untuk FaceNet
    face_crop = Image.fromarray(face_crop)
    face_crop = face_crop.resize((160,160))
    face_crop = np.asarray(face_crop).astype(np.float32)

    face_crop = (face_crop - 127.5) / 128.0
    face_tensor = torch.tensor(face_crop).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face_tensor)
        
    logits = classifier(emb)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs).item()
    conf = probs[0][pred_idx].item()

    name = label_map.get(pred_idx, "Unknown")

    return name, conf


st.set_page_config(page_title="Face Recognition | FaceNet", layout="wide")

st.title("FaceNet for Face Recognition (kelompok 6)")
st.write("Upload photo dan sistem akan mendeteksi identitas mahasiswa")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    name, conf = predict(img)

    with col2:
        if name == "No Face Detected":
            st.error(" No face detected!")
        else:
            st.success(f"**Predicted: {name}**")
            # st.info(f"Confidence: **{conf:.4f}**")
