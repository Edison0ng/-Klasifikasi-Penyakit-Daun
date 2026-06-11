# 🍃 Klasifikasi Penyakit Daun (Plant Disease Classification)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📖 Deskripsi Proyek
Repositori ini berisi aplikasi berbasis antarmuka grafis (GUI) menggunakan **Tkinter** untuk mendeteksi dan mengklasifikasikan penyakit pada daun tanaman. Sistem ini memanfaatkan teknik pengolahan citra digital berupa **Gray Level Co-occurrence Matrix (GLCM)** untuk ekstraksi fitur tekstur, dan menggunakan algoritma **Logistic Regression** dari pustaka *scikit-learn* sebagai model klasifikasi utamanya.

Aplikasi dirancang menggunakan *pipeline* terpadu yang mencakup:
1. **Prapemrosesan Citra:** Konversi ke *grayscale*, perubahan ukuran (*resizing*), pengurangan *noise* (*Gaussian Blur*), normalisasi intensitas, dan kuantisasi level intensitas (16 level).
2. **Ekstraksi Fitur:** Perhitungan otomatis untuk 7 fitur GLCM (*Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM, dan Entropy*).
3. **Pelatihan & Evaluasi Model:** Pembagian data secara otomatis, standardisasi fitur, pembuatan grafik *Confusion Matrix*, serta penyimpanan model terlatih secara lokal.

## 👥 Anggota Kelompok 3
| Nama | NIM |
| :--- | :--- |
| **Joyrich Immanuel Lantang** | 2408561046 |
| **Rahmad Dhani** | 2408561064 |
| **I Wayan Karuna Putra** | 2408561091 |
| **Elijah Maverick Pangau** | 2408561111 |
| **Satrio Palupi** | 2408561138 |

## 📊 Dataset
Dataset yang digunakan diambil langsung dari platform Kaggle, yang mengelompokkan ribuan citra daun berdasarkan jenis tanaman serta kategori penyakitnya.
* 🔗 **Tautan Dataset:** [Plant Disease by emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🛠️ Teknologi & Pustaka Utama
* **Bahasa Pemrograman:** Python
* **Antarmuka Grafis (GUI):** Tkinter
* **Pengolahan Citra & Fitur:** OpenCV, Scikit-Image
* **Pemrosesan & Manajemen Data:** NumPy, Pandas
* **Pemodelan Machine Learning:** Scikit-Learn
* **Visualisasi Data:** Matplotlib, Seaborn
* **Penyimpanan Model:** Joblib

## 🚀 Cara Menjalankan Proyek

1. **Clone repositori ini ke dalam direktori lokal komputer:**
   ```bash
   git clone [https://github.com/Edison0ng/-Klasifikasi-Penyakit-Daun.git](https://github.com/Edison0ng/-Klasifikasi-Penyakit-Daun.git)
   cd -Klasifikasi-Penyakit-Daun 

2. Buat dan aktifkan virtual environment
   ```bash
   python -m venv env
   source env/bin/activate  # Untuk pengguna Linux/Mac
   env\Scripts\activate     # Untuk pengguna Windows
   
3. Instal seluruh dependensi yang terdaftar di requirements.txt
   ```bash
   pip install -r requirements.txt
   
4. Jalankan aplikasi utama
   ```bash
   python APP_Gabungan.py
   
