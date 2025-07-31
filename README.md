# Klasifikasi Multi-Kelas Ikan Menggunakan Transfer Learning dengan ResNet50 dan Optimasi Adam: Analisis Performa dan Interpretabilitas Model melalui Grad-CAM

<img width="1584" height="396" alt="banner" src="https://github.com/user-attachments/assets/f046e4ed-7bcd-4aa0-b4ec-c8b6afd56090" />

Aplikasi web berbasis deep learning yang mengklasifikasikan spesies ikan dengan visualisasi AI yang dapat diinterpretasikan. Dibangun menggunakan Streamlit dan TensorFlow.

## Fitur

- 🐟 **Pengenalan 9 Jenis Ikan**: Mengidentifikasi Black Sea Sprat, Gilt-Head Bream, Horse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, dan Trout
- 📊 **3 Prediksi Teratas**: Menampilkan 3 hasil prediksi terbaik dengan tingkat keyakinan
- 🔍 **Visualisasi AI**: Menggunakan Grad-CAM untuk menyoroti area yang memengaruhi prediksi
- 🚀 **Antarmuka Ramah Pengguna**: Unggah gambar sederhana dengan drag-and-drop
- ☁️ **Loading Model dari Cloud**: Mengunduh model dari Google Drive saat pertama kali dijalankan

## Cara Kerja

Aplikasi ini menggunakan model deep learning ResNet50 yang telah disesuaikan untuk klasifikasi gambar. Komponen teknis utama:

1. **Pemrosesan Gambar**: 
   - Mengubah ukuran dan memproses gambar untuk ResNet50
   - Mendukung berbagai format gambar (JPG, PNG, JPEG)
   
2. **Inferensi Model**:
   - Mengunduh model yang telah dilatih dari Google Drive
   - Melakukan prediksi menggunakan TensorFlow/Keras
   
3. **AI yang Dapat Dijelaskan**:
   - Menghasilkan peta panas Grad-CAM untuk memvisualisasikan area penting
   - Beralih ke peta saliensi jika Grad-CAM gagal
   - Menunjukkan bagian gambar mana yang memengaruhi prediksi

## Demo Langsung

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

## Instalasi

Untuk menjalankan secara lokal:

1. Clone repositori:
```bash
git clone https://github.com/your-username/fish-classification-app.git
cd fish-classification-app
