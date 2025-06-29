
# Emotion Detection Using SVM and CNN

## Deskripsi Proyek

Deteksi emosi (Emotion Detection) merupakan salah satu bidang penerapan dari teknologi Computer Vision yang berfokus pada pengenalan ekspresi wajah manusia melalui gambar atau citra digital. Tujuan utama dari sistem ini adalah untuk memahami kondisi emosional seseorang — seperti marah, senang, sedih, takut, terkejut, netral, hingga jijik — hanya dengan menganalisis ekspresi wajahnya.

Dalam proyek ini, digunakan pendekatan **Support Vector Machine (SVM)** dan **Convolutional Neural Network (CNN)** sebagai metode klasifikasi utama. SVM dikenal efektif untuk klasifikasi dengan data berdimensi tinggi, sedangkan CNN mampu mengenali fitur spasial dari gambar dengan sangat baik. 

## Teknologi yang Digunakan

- Python
- OpenCV
- scikit-learn
- TensorFlow / Keras
- Numpy, Pandas, Matplotlib
- Dataset: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Dataset

Dataset yang digunakan adalah **FER-2013 (Facial Expression Recognition 2013)**, yang terdiri dari lebih dari 35.000 gambar wajah grayscale beresolusi 48x48 piksel. Gambar diklasifikasikan ke dalam 7 emosi:

- 0 = Angry
- 1 = Disgust
- 2 = Fear
- 3 = Happy
- 4 = Sad
- 5 = Surprise
- 6 = Neutral

Dataset tersedia dalam format CSV, di mana tiap baris menyimpan label emosi, data piksel, dan penggunaan data (Training/PublicTest/PrivateTest).

## Alur Pengerjaan

### 1. Pengumpulan Dataset

- Dataset diunduh dari Kaggle.
- File CSV diparsing dan dikonversi menjadi array 48x48 piksel.

### 2. Preprocessing Dataset

- **Reshaping**: Data diubah menjadi bentuk 2D grayscale (48x48).
- **Normalisasi**: Nilai piksel dinormalisasi ke rentang [0, 1].
- **One-Hot Encoding**: Label dikonversi ke format vektor biner.
- **(Opsional)** Augmentasi citra: rotasi, flipping, zooming.

### 3. Pembangunan Model

#### CNN

- **Convolutional Layer** + ReLU + MaxPooling
- **Dropout Layer** untuk menghindari overfitting
- **Fully Connected Layer** untuk klasifikasi akhir

#### SVM

- Ekstraksi fitur menggunakan HOG atau metode lain
- Vektorisasi citra wajah dan training model SVM untuk klasifikasi emosi

### 4. Training & Testing

- Dataset dibagi menjadi training, validation, dan testing set.
- Model dilatih menggunakan optimizer **Adam** dan **loss function categorical crossentropy**.
- Evaluasi model dilakukan melalui:
  - Akurasi
  - Confusion matrix
  - Visualisasi hasil prediksi

## Cara Menjalankan

1. Clone repositori:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   python main.py
   ```

## Potensi Aplikasi

- **Kesehatan**: Membantu diagnosa kondisi emosional pasien.
- **Customer Service**: Mengukur kepuasan pelanggan dari ekspresi.
- **Keamanan (Smart Surveillance)**: Mendeteksi ekspresi mencurigakan.
- **HCI (Human-Computer Interaction)**: Meningkatkan interaksi yang lebih empatik antara manusia dan mesin.

## Hasil & Evaluasi

*(Bagian ini dapat diisi setelah model selesai dilatih dan diuji)*

- Akurasi:
- Confusion Matrix:
- Visualisasi:

---

## Lisensi

Proyek ini berada di bawah lisensi [MIT](LICENSE).
