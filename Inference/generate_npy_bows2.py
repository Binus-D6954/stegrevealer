import os
import numpy as np
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Konfigurasi
# =============================================================================
# Path dataset PNG (Source)
BASE_DIR = r"..\Dataset\Dataset Split RAW BOWS2"

# Path dataset NPY (Output)
OUTPUT_BASE = r"..\Dataset\Dataset Split RAW BOWS2 - NPY"

SPLITS = ['train', 'val', 'test']

# Pemetaan Kelas (Label)
# 0: CLEAN, 1: LSB, 2: PVD, 3: BPCS
CLASS_MAP = {
    'cover': 0,
    'lsb': 1,
    'pvd': 2,
    'bpcs': 3
}

# =============================================================================
# Fungsi Helper
# =============================================================================
def load_single_image(args):
    """Fungsi untuk memuat satu gambar (digunakan oleh thread)."""
    img_path, label = args
    try:
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img, dtype=np.uint8)
        return img_arr, label
    except Exception as e:
        print(f"❌ Error memuat {img_path}: {e}")
        return None

def load_and_convert_parallel(split_name, payload_label):
    """
    Memuat gambar secara paralel menggunakan ThreadPoolExecutor.
    """
    images_list = []
    labels_list = []
    
    split_path = os.path.join(BASE_DIR, split_name)
    
    # Kelas yang akan dimuat
    classes_to_load = [
        ('cover', 'cover'),
        (f'stego_lsb_{payload_label}', 'lsb'),
        (f'stego_pvd_{payload_label}', 'pvd'),
        (f'stego_bpcs_{payload_label}', 'bpcs')
    ]
    
    print(f"📂 [VERBOSE] Memulai pemrosesan split: {split_name.upper()} (Payload: 0.{payload_label})")
    
    all_tasks = []
    for folder_name, cls_key in classes_to_load:
        folder_path = os.path.join(split_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️  [VERBOSE] Folder tidak ditemukan: {folder_path}")
            continue
            
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        label = CLASS_MAP[cls_key]
        
        print(f"   🔍 [VERBOSE] Mengantre {len(files)} gambar dari folder '{folder_name}'...")
        for f in files:
            all_tasks.append((os.path.join(folder_path, f), label))
            
    total_files = len(all_tasks)
    if total_files == 0:
        return None, None

    # Eksekusi Paralel
    print(f"   ⚡ [VERBOSE] Membaca {total_files} file menggunakan multi-threading...")
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = list(executor.map(load_single_image, all_tasks))
        
        # Filter None dan pisahkan hasil
        for i, res in enumerate(results):
            if res:
                images_list.append(res[0])
                labels_list.append(res[1])
            
            # Tampilkan progres setiap 1000 gambar
            if (i + 1) % 1000 == 0:
                print(f"   ⏳ [VERBOSE] Progres: {i + 1}/{total_files} gambar termuat...")

    return np.array(images_list, dtype=np.uint8), np.array(labels_list, dtype=np.uint8)

def save_npy(payload_label):
    # Nama folder sesuai format: "0.1 NPY" atau "0.4 NPY"
    if payload_label == "01":
        payload_dir = os.path.join(OUTPUT_BASE, "0.1 NPY")
    elif payload_label == "04":
        payload_dir = os.path.join(OUTPUT_BASE, "0.4 NPY")
    else:
        payload_dir = os.path.join(OUTPUT_BASE, f"0.{payload_label} NPY")
        
    os.makedirs(payload_dir, exist_ok=True)
    
    for split in SPLITS:
        print("-" * 50)
        start_time = time.time()
        
        imgs, lbls = load_and_convert_parallel(split, payload_label)
        
        if imgs is None or len(imgs) == 0:
            print(f"❌ [VERBOSE] Tidak ada data ditemukan untuk {split} 0.{payload_label}")
            continue
            
        img_out = os.path.join(payload_dir, f"{split}_images.npy")
        lbl_out = os.path.join(payload_dir, f"{split}_labels.npy")
        
        print(f"   💾 [VERBOSE] Menyimpan ke: {img_out}...")
        np.save(img_out, imgs)
        print(f"   💾 [VERBOSE] Menyimpan ke: {lbl_out}...")
        np.save(lbl_out, lbls)
        
        elapsed = time.time() - start_time
        print(f"✅ [VERBOSE] Split {split} selesai dalam {elapsed:.2f} detik.")

if __name__ == "__main__":
    print("="*70)
    print("🚀 NPY GENERATOR - BOWS2 (Multi-threaded & Verbose)")
    print("="*70)
    print(f"📁 Source Dir: {os.path.abspath(BASE_DIR)}")
    print(f"📂 Output Dir: {os.path.abspath(OUTPUT_BASE)}\n")
    
    total_start = time.time()
    
    save_npy("01")
    print("\n" + "*"*70 + "\n")
    save_npy("04")
    
    total_elapsed = time.time() - total_start
    print("="*70)
    print(f"🎉 SEMUA DATASET NPY BERHASIL DIBUAT!")
    print(f"⏱️ Total waktu eksekusi: {total_elapsed/60:.2f} menit.")
    print(f"📂 Lokasi: {OUTPUT_BASE}")
    print("="*70)
