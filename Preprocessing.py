import os
import cv2
from tqdm import tqdm 
from collections import defaultdict

def preprocessing_citra(img, crop_percent=0.8):
    """Fungsi inti: Crop, Grayscale, dan Gaussian Blur"""
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    img_crop = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    return img_blur

def proses_folder_ke_folder(input_folder_path, output_folder_name):
    print("=" * 60)
    print("📁 PREPROCESSING DATASET GAMBAR → FOLDER")
    print("=" * 60)
    print(f"[*] Folder Input: {input_folder_path}")
    print(f"[*] Folder Output: {output_folder_name}/")
    print("-" * 60)
    
    if not os.path.exists(input_folder_path):
        print(f"[X] ERROR: Folder '{input_folder_path}' tidak ditemukan!")
        return False
    
    # Hitung total dan list file
    print("[*] Menghitung total gambar...")
    daftar_file = []
    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                daftar_file.append((root, file))
    
    total_gambar = len(daftar_file)
    if total_gambar == 0:
        print("[X] Tidak ada gambar ditemukan!")
        return False
    
    print("[*] Memulai preprocessing ke folder...\n")
    
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
        print(f"[✓] Folder output dibuat: {output_folder_name}/")
    
    # Dictionary untuk menghitung data per folder/kelas
    stats_per_folder = defaultdict(int)
    jumlah_diproses = 0
    gambar_gagal = 0
    
    for root, file in tqdm(daftar_file, desc="Processing", unit="gambar"):
        path_asli = os.path.join(root, file)
        img = cv2.imread(path_asli)
        
        if img is None:
            gambar_gagal += 1
            continue
        
        try:
            img_bersih = preprocessing_citra(img)
            rel_path = os.path.relpath(root, input_folder_path)
            output_dir = os.path.join(output_folder_name, rel_path) if rel_path != '.' else output_folder_name
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, file)
            success = cv2.imwrite(output_path, img_bersih)
            
            if success:
                jumlah_diproses += 1
                # Simpan statistik berdasarkan nama sub-folder (kelas)
                nama_kelas = rel_path if rel_path != '.' else "Root"
                stats_per_folder[nama_kelas] += 1
            else:
                gambar_gagal += 1
                
        except Exception as e:
            gambar_gagal += 1

    # Menampilkan ringkasan akhir
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING SELESAI!")
    print("=" * 60)
    print(f"{'Nama Folder/Kelas':<40} | {'Jumlah':<10}")
    print("-" * 60)
    for folder, count in sorted(stats_per_folder.items()):
        print(f"{folder:<40} | {count:<10} gambar")
    print("-" * 60)
    print(f"📈 Total Berhasil: {jumlah_diproses} | ⚠️ Gagal: {gambar_gagal}")
    print(f"📁 Hasil tersimpan di: '{output_folder_name}/'")
    print("=" * 60)

if __name__ == "__main__":
    FOLDER_INPUT = "input"
    OUTPUT_FOLDER = "dataset_bersih2"
    proses_folder_ke_folder(FOLDER_INPUT, OUTPUT_FOLDER)