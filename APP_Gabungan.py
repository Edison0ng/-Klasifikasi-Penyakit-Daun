import os
import cv2
import numpy as np
import pandas as pd
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import joblib

# Library untuk Machine Learning & Plotting
from skimage.feature import graycomatrix, graycoprops
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Supaya aman dijalankan di background thread Tkinter

class AplikasiKlasifikasiTanaman:
    def __init__(self, root):
        self.root = root
        self.root.title("🍃 Sistem Deteksi Penyakit Daun (GLCM & Logistic Regression)")
        self.root.geometry("750x680") 
        self.root.configure(bg="#f8f9fa")
        
        self.folder_dataset = "" # DIUBAH: Menyimpan path folder, bukan zip
        self.model_tersedia = False
        self.scaler = None
        self.model = None
        self.label_encoder = None
        
        self.buat_ui()

    def buat_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        tabControl = ttk.Notebook(self.root)
        self.tab_train = ttk.Frame(tabControl)
        self.tab_prediksi = ttk.Frame(tabControl)
        
        tabControl.add(self.tab_train, text='⚙️ Training Model (Pipeline)')
        tabControl.add(self.tab_prediksi, text='🔍 Prediksi Gambar Baru')
        tabControl.pack(expand=1, fill="both", padx=10, pady=10)

        self._setup_tab_train()
        self._setup_tab_prediksi()

    def _setup_tab_train(self):
        frame_header = tk.Frame(self.tab_train, bg="#2c3e50", pady=15)
        frame_header.pack(fill=tk.X)
        tk.Label(frame_header, text="Pipeline Terpadu: Preprocessing ➔ GLCM ➔ Logistic Regression", 
                 font=("Segoe UI", 12, "bold"), fg="white", bg="#2c3e50").pack()

        frame_kontrol = tk.Frame(self.tab_train, bg="#f8f9fa", pady=15, padx=15)
        frame_kontrol.pack(fill=tk.X)

        self.btn_pilih = ttk.Button(frame_kontrol, text="📁 1. Pilih Folder Dataset", command=self.pilih_folder)
        self.btn_pilih.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.lbl_path = tk.Label(frame_kontrol, text="Belum ada folder yang dipilih", bg="#f8f9fa", fg="#7f8c8d")
        self.lbl_path.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        tk.Label(frame_kontrol, text="🏷️ Nama Folder Output:", bg="#f8f9fa", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.entry_output = ttk.Entry(frame_kontrol, width=30)
        self.entry_output.insert(0, "Skenario_1") 
        self.entry_output.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.btn_mulai = ttk.Button(frame_kontrol, text="▶️ 2. Mulai Seluruh Proses", command=self.mulai_pipeline, state=tk.DISABLED)
        self.btn_mulai.grid(row=2, column=0, columnspan=2, pady=15, sticky="we", padx=5)

        self.progress = ttk.Progressbar(frame_kontrol, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky="we", padx=5)

        frame_log = tk.Frame(self.tab_train, bg="#f8f9fa", padx=15, pady=5)
        frame_log.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame_log, text="Terminal Status:", font=("Segoe UI", 9, "bold"), bg="#f8f9fa").pack(anchor="w")
        scrollbar = tk.Scrollbar(frame_log)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log = tk.Text(frame_log, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_log.yview)

    def _setup_tab_prediksi(self):
        frame_pred = tk.Frame(self.tab_prediksi, bg="#f8f9fa", padx=20, pady=20)
        frame_pred.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame_pred, text="Uji Model dengan Gambar Baru", font=("Segoe UI", 14, "bold"), bg="#f8f9fa").pack(pady=10)
        
        self.btn_uji = ttk.Button(frame_pred, text="🖼️ Pilih Gambar Daun", command=self.prediksi_gambar_baru)
        self.btn_uji.pack(pady=10)
        
        self.lbl_hasil_prediksi = tk.Label(frame_pred, text="-", font=("Segoe UI", 16), bg="#f8f9fa", fg="#c0392b")
        self.lbl_hasil_prediksi.pack(pady=20)

    def cetak_log(self, pesan):
        self.root.after(0, self._tulis_ke_layar, pesan)

    def _tulis_ke_layar(self, pesan):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, pesan + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def pilih_folder(self):
        folder_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if folder_path:
            self.folder_dataset = folder_path
            self.lbl_path.config(text=f".../{os.path.basename(folder_path)}")
            self.btn_mulai.config(state=tk.NORMAL)
            self.cetak_log(f"[*] Folder dipilih: {os.path.basename(folder_path)}")

    def mulai_pipeline(self):
        self.btn_pilih.config(state=tk.DISABLED)
        self.btn_mulai.config(state=tk.DISABLED)
        self.entry_output.config(state=tk.DISABLED)
        self.progress['value'] = 0
        thread = threading.Thread(target=self.proses_semua)
        thread.daemon = True
        thread.start()

    def preprocessing_citra(self, img, target_size=(256, 256), glcm_levels=16):
        # 1. Konversi ke Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resizing Citra (Memaksa ukuran menjadi sama, misalnya 256x256)
        img_resized = cv2.resize(img_gray, target_size)
        
        # Blur untuk mengurangi noise
        img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
        
        # 3. Normalisasi Intensitas (Meratakan kontras ke rentang 0-255)
        img_norm = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX)
        
        # 4. Kuantisasi Level Intensitas (256 -> 16 level)
        bin_size = 256 // glcm_levels
        img_quantized = (img_norm // bin_size).astype(np.uint8)
        
        return img_quantized

    def hitung_entropy_glcm(self, glcm):
        p = glcm / np.sum(glcm)
        p = p[p > 0] 
        return -np.sum(p * np.log2(p))

    def proses_semua(self):
        data_fitur = []
        
        nama_custom = self.entry_output.get().strip()
        if not nama_custom:
            nama_custom = "Default"
            
        folder_laporan = f"Output_{nama_custom}"
        folder_prep = os.path.join(folder_laporan, "1_Hasil_Preprocessing")
        os.makedirs(folder_prep, exist_ok=True)
        
        try:
            self.cetak_log("\n" + "="*50)
            self.cetak_log(f"[*] Folder output disetel ke: {folder_laporan}/")
            
            self.cetak_log("[1/4] Membaca Folder Dataset...")
            self.progress['value'] = 10
            
            self.cetak_log("[2/4] Preprocessing & Ekstraksi Fitur GLCM...")
            daftar_gambar = []
            
            for root, dirs, files in os.walk(self.folder_dataset):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        daftar_gambar.append(os.path.join(root, f))
            
            total_img = len(daftar_gambar)
            
            if total_img == 0:
                raise Exception("Tidak ada gambar (.png, .jpg, .jpeg) di dalam folder tersebut!")
            
            for idx, path_file in enumerate(daftar_gambar):
                kelas_folder = os.path.basename(os.path.dirname(path_file))
                nama_file = os.path.basename(path_file)
                
                img_raw = cv2.imread(path_file)
                if img_raw is None: continue
                
                # --- PROSES PREPROCESSING ---
                img_bersih = self.preprocessing_citra(img_raw)
                
                folder_kelas_out = os.path.join(folder_prep, kelas_folder)
                os.makedirs(folder_kelas_out, exist_ok=True)
                cv2.imwrite(os.path.join(folder_kelas_out, nama_file), img_bersih)
                
                # --- PROSES GLCM ---
                glcm = graycomatrix(img_bersih, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    levels=16, symmetric=True, normed=True)
                
                # --- FITUR BARU: SIMPAN GRAFIK MATRIKS GLCM (Hanya untuk gambar pertama) ---
                if idx == 0:
                    self.cetak_log("      Menyimpan Grafik Matriks GLCM Preprocessing...")
                    plt.figure(figsize=(8, 6))
                    # Mengambil slice matriks GLCM untuk jarak 1 dan sudut 0 derajat
                    sns.heatmap(glcm[:, :, 0, 0], cmap="Blues", annot=False)
                    plt.title(f"Visualisasi Matriks GLCM 16x16 (Hasil Kuantisasi)\nFile: {nama_file}")
                    plt.xlabel("Intensitas Piksel Tetangga (Level 0-15)")
                    plt.ylabel("Intensitas Piksel Referensi (Level 0-15)")
                    plt.tight_layout()
                    
                    path_grafik_glcm = os.path.join(folder_laporan, f"1a_Matriks_GLCM_Preproc_{nama_custom}.png")
                    plt.savefig(path_grafik_glcm, dpi=150)
                    plt.close()
                # -------------------------------------------------------------------------

                contrast = graycoprops(glcm, 'contrast').mean()
                energy = graycoprops(glcm, 'energy').mean()
                homogeneity = graycoprops(glcm, 'homogeneity').mean()
                correlation = graycoprops(glcm, 'correlation').mean()
                dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
                asm = graycoprops(glcm, 'ASM').mean()
                entropy = self.hitung_entropy_glcm(glcm)
                
                data_fitur.append([nama_file, kelas_folder, contrast, energy, homogeneity, correlation, dissimilarity, asm, entropy])
                
                if (idx+1) % 50 == 0:
                    self.cetak_log(f"      Mengekstrak: {idx+1}/{total_img} gambar...")
                
                self.root.after(0, lambda v=(10 + (idx/total_img)*40): self.progress.config(value=v))

            self.cetak_log("[3/4] Menyimpan data fitur ke CSV...")
            kolom = ['Nama_File', 'Kelas', 'Contrast', 'Energy', 'Homogeneity', 'Correlation', 'Dissimilarity', 'ASM', 'Entropy']
            df = pd.DataFrame(data_fitur, columns=kolom)
            
            path_csv_glcm = os.path.join(folder_laporan, f"2_Dataset_Fitur_{nama_custom}.csv")
            df.to_csv(path_csv_glcm, index=False)
            self.progress['value'] = 60

            self.cetak_log("[4/4] Melatih Model Logistic Regression...")
            fitur_kolom = ["Contrast", "Energy", "Homogeneity", "Correlation", "Dissimilarity", "ASM", "Entropy"]
            X = df[fitur_kolom]
            y = df["Kelas"]

            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
            
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.cetak_log(f"      [✓] Akurasi Model: {acc*100:.2f}%")
            
            df_test = pd.DataFrame(self.scaler.inverse_transform(X_test), columns=fitur_kolom)
            df_test["Label_Asli"] = self.label_encoder.inverse_transform(y_test)
            df_test["Label_Prediksi"] = self.label_encoder.inverse_transform(y_pred)
            df_test["Status_Prediksi"] = np.where(df_test["Label_Asli"] == df_test["Label_Prediksi"], "BENAR", "SALAH")
            
            path_csv_test = os.path.join(folder_laporan, f"3_Detail_Prediksi_{nama_custom}.csv")
            df_test.to_csv(path_csv_test, index=False)

            self.cetak_log("      Menyimpan Visualisasi (Confusion Matrix)...")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            plt.title(f"Confusion Matrix ({nama_custom})")
            plt.ylabel("Aktual")
            plt.xlabel("Prediksi")
            plt.tight_layout()
            
            path_grafik = os.path.join(folder_laporan, f"4_CM_{nama_custom}.png")
            plt.savefig(path_grafik, dpi=150)
            plt.close()

            joblib.dump(self.model, os.path.join(folder_laporan, f'model_{nama_custom}.pkl'))
            joblib.dump(self.scaler, os.path.join(folder_laporan, f'scaler_{nama_custom}.pkl'))
            joblib.dump(self.label_encoder, os.path.join(folder_laporan, f'le_{nama_custom}.pkl'))
            
            self.progress['value'] = 100
            self.model_tersedia = True
            
            self.cetak_log("="*50)
            self.cetak_log("🎉 SELURUH PROSES SELESAI!")
            self.cetak_log(f"Semua hasil disimpan rapi di folder '{folder_laporan}'.")
            self.root.after(0, lambda: messagebox.showinfo("Selesai", f"Berhasil!\nAkurasi Model: {acc*100:.2f}%\nFolder '{folder_laporan}' telah dibuat."))

        except Exception as e:
            self.cetak_log(f"\n[X] ERROR: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Terjadi Kesalahan:\n{str(e)}"))
            
        finally:
            self.cetak_log("[*] Proses dihentikan/selesai.")
            self.root.after(0, lambda: self.btn_pilih.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_mulai.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.entry_output.config(state=tk.NORMAL))

    def prediksi_gambar_baru(self):
        if not self.model_tersedia:
            try:
                nama = self.entry_output.get().strip() or "Default"
                folder_laporan = f"Output_{nama}"
                self.model = joblib.load(os.path.join(folder_laporan, f'model_{nama}.pkl'))
                self.scaler = joblib.load(os.path.join(folder_laporan, f'scaler_{nama}.pkl'))
                self.label_encoder = joblib.load(os.path.join(folder_laporan, f'le_{nama}.pkl'))
                self.model_tersedia = True
            except:
                messagebox.showwarning("Peringatan", "Model belum dilatih atau folder tidak ditemukan! Lakukan Training dulu.")
                return

        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path: return
        
        try:
            img_raw = cv2.imread(file_path)
            img_bersih = self.preprocessing_citra(img_raw)
            
            glcm = graycomatrix(img_bersih, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                levels=16, symmetric=True, normed=True)
            fitur = [
                graycoprops(glcm, 'contrast').mean(),
                graycoprops(glcm, 'energy').mean(),
                graycoprops(glcm, 'homogeneity').mean(),
                graycoprops(glcm, 'correlation').mean(),
                graycoprops(glcm, 'dissimilarity').mean(),
                graycoprops(glcm, 'ASM').mean(),
                self.hitung_entropy_glcm(glcm)
            ]
            
            fitur_scaled = self.scaler.transform([fitur])
            pred_idx = self.model.predict(fitur_scaled)[0]
            probabilitas = self.model.predict_proba(fitur_scaled).max() * 100
            
            kelas_prediksi = self.label_encoder.inverse_transform([pred_idx])[0]
            
            self.lbl_hasil_prediksi.config(text=f"Hasil: {kelas_prediksi.upper()}\n(Kepercayaan: {probabilitas:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memprediksi gambar:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiKlasifikasiTanaman(root)
    root.mainloop()