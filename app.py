import streamlit as st
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
from PIL import Image
import time

# Load model
model = load_model("model_ekspresi.h5")
labels = ['marah', 'jijik', 'takut', 'senang', 'sedih', 'kaget', 'netral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# UI
st.title("😊 Deteksi Ekspresi Wajah")
mode = st.radio("Pilih metode input:", ["📂 Upload Gambar", "📷 Kamera Real-Time"])

FRAME_WINDOW = st.empty()

def proses_gambar(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ekspresi_counter = {}

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        preds = model.predict(face)
        idx = np.argmax(preds)
        label = labels[idx]
        confidence = preds[0][idx]

        ekspresi_counter[label] = ekspresi_counter.get(label, 0) + 1

        # Tampilkan di gambar
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_np, f"{label} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return image_np, ekspresi_counter

if mode == "📂 Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        hasil, ekspresi_counter = proses_gambar(image_np)
        st.image(hasil, caption="Hasil Deteksi", use_container_width=True)

        if ekspresi_counter:
            st.subheader("📊 Diagram Ekspresi Terdeteksi")
            df = pd.DataFrame.from_dict(ekspresi_counter, orient='index', columns=['Jumlah'])
            st.bar_chart(df)

elif mode == "📷 Kamera Real-Time":
    start = st.checkbox("🎬 Aktifkan Kamera")

    if start:
        cap = cv2.VideoCapture(0)

        while start:
            ret, frame = cap.read()
            if not ret:
                st.warning("Tidak bisa membaca kamera.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hasil, ekspresi_counter = proses_gambar(frame_rgb)
            FRAME_WINDOW.image(hasil, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
