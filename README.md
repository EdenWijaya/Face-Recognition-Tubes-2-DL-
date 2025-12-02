# 🎓 Face Recognition Model Comparison
_Perbandingan FaceNet (CNN) dan Swin Transformer (Vision Transformer) untuk Sistem Presensi Berbasis Pengenalan Wajah_

Repository ini berisi eksperimen pengembangan sistem presensi otomatis berbasis Face Recognition dengan membandingkan dua model Deep Learning:

- **FaceNet** (CNN-based, face embedding)
- **Swin Transformer** (Vision Transformer-based)

Seluruh proses dilakukan melalui **Jupyter Notebook (`.ipynb`)**, mencakup:
- Dataset loading
- Preprocessing wajah
- Training & evaluation
- Perbandingan metrik performa

> Model terbaik ditentukan berdasarkan hasil akurasi, stabilitas, dan kemampuan generalisasi terhadap dataset wajah mahasiswa.

---

# 🗂 Dataset

Dataset berisi foto wajah mahasiswa.  
Setiap mahasiswa direpresentasikan sebagai **satu kelas (identity)** dan memiliki folder masing-masing.

## Struktur Dataset
```bash
dataset/
└── train/
    ├── Nama_Siswa_A/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── img3.jpg
    │   └── img4.jpg
    ├── Nama_Siswa_B/
    ├── Nama_Siswa_C/
    └── ...
└── val/
    ├── Nama_Siswa_A/
    ├── Nama_Siswa_B/
    └── ...
```
# 🚀 Cara Menjalankan Notebook

## 1. Clone Repository
```bash
git clone <link_repository>
cd <folder_repository>
```
## 2. Install Dependencies
```bash
pip install torch torchvision timm facenet-pytorch sklearn numpy matplotlib
```

## 3. Struktur Folder Wajib
```bash
dataset/
  train/
  val/
```

## 4. Set Path Dataset Pada Notebook
```bash
DATA_ROOT = r".../data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
```

Hasil Eksperimen
| Model            | Best Val Accuracy |
| ---------------- | ----------------- |
| **FaceNet**      | **74%**           |
| Swin Transformer | 61%               |


### 👨‍💻 Authors

Kelompok 5 
Deep Learning Project
Face Recognition Attendance System

- Eden Wijaya
- 12214020 - Intan Permata Sari
- Bayu Ega Ferdana
