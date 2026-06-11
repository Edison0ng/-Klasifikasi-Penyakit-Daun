import os
import cv2
import numpy as np
import pandas as pd
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage.feature import graycomatrix, graycoprops
import zipfile
import shutil

class AplikasiGLCM:
    def __init__(self, root):
        self.root = root
        self.root.title("🍃 Ekstraksi Fitur GLCM - Penyakit Daun")
        self.root.geometry("650x550")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(False, False)

        self.file_zip_dataset = ""
        self.sedang_proses = False

        self.buat_ui()

    def buat_ui(self):
        # Frame Judul
        frame_judul = tk.Frame(self.root, bg="#2c3e50", pady=15)
        frame_judul.pack(fill=tk.X)
        
        lbl_judul = tk.Label(frame_judul, text="Sistem Ekstraksi Fitur GLCM", 
                             font=("Segoe UI", 16, "bold"), fg="white", bg="#2c3e50")
        lbl_judul.pack()
        
        lbl_subjudul = tk.Label(frame_judul, text="Input ZIP Dataset ➔ Output CSV", 
                                font=("Segoe UI", 10), fg="#bdc3c7", bg="#2c3e50")
        lbl_subjudul.pack()

        # Frame Kontrol
        frame_kontrol = tk.Frame(self.root, bg="#f0f4f8", pady=20, padx=20)
        frame_kontrol.pack(fill=tk.X)

        # UBAH: Sekarang tombol ini mencari file ZIP
        self.btn_pilih = ttk.Button(frame_kontrol, text="📦 Pilih File ZIP Dataset", command=self.pilih_zip)
        self.btn_pilih.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.lbl_path = tk.Label(frame_kontrol, text="Belum ada file ZIP yang dipilih", 
                                 font=("Segoe UI", 9, "italic"), bg="#f0f4f8", fg="#7f8c8d")
        self.lbl_path.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.btn_mulai = ttk.Button(frame_kontrol, text="▶️ Mulai Ekstraksi", command=self.mulai_proses, state=tk.DISABLED)
        self.btn_mulai.grid(row=1, column=0, columnspan=2, pady=15, sticky="we")

        # Frame Log/Terminal Visual
        frame_log = tk.Frame(self.root, bg="#f0f4f8", padx=20, pady=5)
        frame_log.pack(fill=tk.BOTH, expand=True)

        lbl_log = tk.Label(frame_log, text="Status Proses:", font=("Segoe UI", 10, "bold"), bg="#f0f4f8")
        lbl_log.pack(anchor="w")

        # Scrollbar dan Text Area
        scrollbar = tk.Scrollbar(frame_log)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_log = tk.Text(frame_log, height=15, bg="#1e1e1e", fg="#00ff00", 
                               font=("Consolas", 9), yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_log.yview)

    def cetak_log(self, pesan):
        self.root.after(0, self._tulis_ke_layar, pesan)

    def _tulis_ke_layar(self, pesan):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, pesan + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def pilih_zip(self):
        # Dialog hanya akan menampilkan file berekstensi .zip
        file_path = filedialog.askopenfilename(
            title="Pilih File ZIP Dataset (misal: dataset_bersih.zip)",
            filetypes=[("ZIP Files", "*.zip")]
        )
        if file_path:
            self.file_zip_dataset = file_path
            self.lbl_path.config(text=f".../{os.path.basename(file_path)}")
            self.btn_mulai.config(state=tk.NORMAL)
            self.cetak_log(f"[*] File ZIP dipilih: {os.path.basename(file_path)}")

    def mulai_proses(self):
        if not self.file_zip_dataset:
            messagebox.showwarning("Peringatan", "Pilih file ZIP dataset terlebih dahulu!")
            return
            
        self.btn_pilih.config(state=tk.DISABLED)
        self.btn_mulai.config(state=tk.DISABLED)
        self.cetak_log("\n" + "="*50)
        self.cetak_log("[START] Memulai ekstraksi GLCM dari ZIP...")
        
        thread = threading.Thread(target=self.proses_ekstraksi)
        thread.daemon = True
        thread.start()

    def proses_ekstraksi(self):
        data_fitur = []
        temp_dir = "temp_glcm_extract"
        
        try:
            # 1. Ekstrak ZIP ke folder sementara
            self.cetak_log(f"[*] Membongkar {os.path.basename(self.file_zip_dataset)} (sementara)...")
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(self.file_zip_dataset, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 2. Cari semua gambar di dalam folder hasil ekstrak
            self.cetak_log("[*] Mencari gambar dan menghitung matriks GLCM...")
            
            jumlah_diproses = 0
            # os.walk memastikan script masuk ke dalam sub-folder sedalam apapun
            for root, dirs, files in os.walk(temp_dir):
                gambar_di_folder = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not gambar_di_folder:
                    continue
                
                # Nama kelas diambil dari nama folder tempat gambar itu berada
                kelas = os.path.basename(root)
                
                # Tentukan label: 0 jika 'healthy', 1 jika sakit
                if 'healthy' in kelas.lower() or 'sehat' in kelas.lower():
                    label = 0
                    status = "SEHAT"
                else:
                    label = 1
                    status = "SAKIT"
                
                self.cetak_log(f"\n[+] Kelas: {kelas} -> Label: {label} ({status})")
                
                # 3. Proses setiap gambar dengan GLCM
                for i, nama_file in enumerate(gambar_di_folder, 1):
                    path_file = os.path.join(root, nama_file)
                    
                    if i % 50 == 0 or i == len(gambar_di_folder):
                        self.cetak_log(f"    -> Memproses: {i}/{len(gambar_di_folder)} gambar")
                    
                    img = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    
                    try:
                        # Hitung GLCM
                        glcm = graycomatrix(img, distances=[1], 
                                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                            levels=256, symmetric=True, normed=True)
                        
                        contrast = graycoprops(glcm, 'contrast').mean()
                        energy = graycoprops(glcm, 'energy').mean()
                        homogeneity = graycoprops(glcm, 'homogeneity').mean()
                        correlation = graycoprops(glcm, 'correlation').mean()
                        
                        data_fitur.append([nama_file, kelas, contrast, energy, homogeneity, correlation, label])
                        jumlah_diproses += 1
                        
                    except Exception as e:
                        pass

            # 4. Simpan ke CSV
            if len(data_fitur) > 0:
                self.cetak_log("\n" + "="*50)
                self.cetak_log("[*] Mengemas hasil menjadi dataset_fitur.csv ...")
                
                df = pd.DataFrame(data_fitur, columns=['Nama_File', 'Kelas', 'Contrast', 'Energy', 'Homogeneity', 'Correlation', 'Label'])
                path_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_fitur.csv')
                df.to_csv(path_csv, index=False)
                
                self.cetak_log(f"[V] BERHASIL! File tersimpan di: {path_csv}")
                self.cetak_log(f"    - Total Data Diekstrak: {jumlah_diproses} gambar")
                self.root.after(0, lambda: messagebox.showinfo("Selesai", f"Ekstraksi GLCM Berhasil!\n{jumlah_diproses} gambar diproses menjadi CSV."))
            else:
                self.cetak_log("\n[X] GAGAL: Tidak ada data valid yang bisa diekstrak dari ZIP tersebut.")
                self.root.after(0, lambda: messagebox.showerror("Gagal", "ZIP kosong atau tidak ada gambar valid!"))

        except Exception as e:
            self.cetak_log(f"[X] Terjadi Kesalahan Kritis: {str(e)}")
            
        finally:
            # 5. BERSIH-BERSIH: Hapus folder sementara agar laptop tidak penuh
            self.cetak_log("[*] Membersihkan file sementara...")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            self._selesai_proses()

    def _selesai_proses(self):
        self.root.after(0, lambda: self.btn_pilih.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_mulai.config(state=tk.NORMAL))
        self.cetak_log("="*50)
        self.cetak_log("[READY] Silakan periksa file CSV atau pilih ZIP lain.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiGLCM(root)
    root.mainloop()