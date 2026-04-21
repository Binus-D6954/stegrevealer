"""
Interactive Inference Script for SRNet Steganalysis
Supports both single-channel (RAW only) and hybrid (RAW+REVEAL) models.
Automatically detects input channels from checkpoint.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# =============================================================================
# Konfigurasi
# =============================================================================
MODEL_DIR = "models"                    # Folder tempat file .pth disimpan
CLASSES = ['CLEAN', 'LSB', 'PVD', 'BPCS']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256

# =============================================================================
# Arsitektur SRNet (dengan parameter in_channels)
# =============================================================================
class SRMLayer(nn.Module):
    def __init__(self, threshold=3.0):
        super().__init__()
        self.T = threshold
    def forward(self, x):
        return x.clamp(-self.T, self.T)

class Type1Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.block(x) + x)

class Type2Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.pool(self.shortcut(x))
        out = self.pool(self.block(x))
        return self.relu(out + residual)

class SRNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.srm = SRMLayer()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.type1_blocks = nn.Sequential(Type1Block(64), Type1Block(64))
        self.type2_blocks = nn.Sequential(
            Type2Block(64, 128),
            Type2Block(128, 256),
            Type2Block(256, 512),
            Type2Block(512, 512),
            Type2Block(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.srm(x)
        x = self.layer1(x)
        x = self.type1_blocks(x)
        x = self.type2_blocks(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

# =============================================================================
# Fungsi Ekstraksi LSB Visual (dalam memori)
# =============================================================================
def extract_lsb_visual_array(input_path):
    """
    Membaca gambar, mengekstrak LSB dari setiap channel RGB,
    mengalikan dengan 255, dan mengembalikan array grayscale (0/255).
    """
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)  # (H, W, 3)

    # Ambil LSB (bit terakhir) dan ubah ke 0 atau 255
    red_lsb   = (img_array[:, :, 0] & 1) * 255
    green_lsb = (img_array[:, :, 1] & 1) * 255
    blue_lsb  = (img_array[:, :, 2] & 1) * 255

    # Konversi ke grayscale dengan luminance
    lsb_gray = (0.299 * red_lsb + 0.587 * green_lsb + 0.114 * blue_lsb).astype(np.uint8)
    return lsb_gray  # shape (H, W), nilai 0 atau 255

# =============================================================================
# Preprocessing
# =============================================================================
def preprocess_single(image_path):
    """Untuk model single‑channel (RAW saja)."""
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    arr = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(DEVICE)

def preprocess_hybrid(image_path):
    """Untuk model hybrid: otomatis ekstrak LSB visual dari gambar RAW."""
    # 1. RAW channel
    raw_img = Image.open(image_path).convert('L')
    raw_img = raw_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    raw_arr = np.array(raw_img, dtype=np.uint8)
    raw_norm = (raw_arr / 127.5) - 1.0

    # 2. REVEAL channel
    reveal_arr = extract_lsb_visual_array(image_path)
    reveal_img = Image.fromarray(reveal_arr)
    reveal_img = reveal_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    reveal_arr = np.array(reveal_img, dtype=np.uint8)
    reveal_norm = (reveal_arr / 127.5) - 1.0

    # 3. Gabungkan: (2, H, W)
    hybrid = np.stack([raw_norm, reveal_norm], axis=0)
    tensor = torch.from_numpy(hybrid).float().unsqueeze(0)  # (1, 2, H, W)
    return tensor.to(DEVICE)

# =============================================================================
# Deteksi jumlah channel dari checkpoint
# =============================================================================
def get_in_channels_from_checkpoint(checkpoint_path):
    """Membaca checkpoint dan mengembalikan jumlah input channel."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Bobot layer pertama: 'layer1.0.weight' shape (out_ch, in_ch, k, k)
    weight = state_dict['layer1.0.weight']
    in_channels = weight.shape[1]
    return in_channels

# =============================================================================
# Load Model
# =============================================================================
def list_model_files(directory):
    if not os.path.exists(directory):
        print(f"❌ Direktori model '{directory}' tidak ditemukan.")
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    return sorted(files)

def load_model(model_path):
    # Deteksi channel
    in_channels = get_in_channels_from_checkpoint(model_path)
    print(f"📌 Model terdeteksi memiliki {in_channels} input channel(s).")

    model = SRNet(num_classes=len(CLASSES), in_channels=in_channels).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)

    # Tangani kemungkinan jumlah kelas berbeda
    fc_weight = state_dict['classifier.weight']
    if fc_weight.shape[0] != len(CLASSES):
        print(f"⚠️  Checkpoint memiliki {fc_weight.shape[0]} kelas, model ini menggunakan {len(CLASSES)} kelas.")
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        print("✅ Bobot fitur ekstraktor berhasil dimuat. Classifier diinisialisasi ulang.")
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model, in_channels

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# =============================================================================
# Program Utama
# =============================================================================
def main():
    print("\n" + "="*70)
    print("🔍 SRNet Steganalysis - Universal Inference")
    print("   Mendukung model Single (RAW) dan Hybrid (RAW+REVEAL)")
    print("="*70)

    # 1. Cek dan pilih model
    model_files = list_model_files(MODEL_DIR)
    if not model_files:
        print(f"Tidak ada file .pth di folder '{MODEL_DIR}'.")
        return

    print("\n📁 Model yang tersedia:")
    for i, fname in enumerate(model_files):
        print(f"  [{i+1}] {fname}")

    try:
        choice = int(input("\n👉 Pilih nomor model: ")) - 1
        if choice < 0 or choice >= len(model_files):
            raise ValueError
    except ValueError:
        print("❌ Pilihan tidak valid.")
        return

    model_path = os.path.join(MODEL_DIR, model_files[choice])
    print(f"\n✅ Model dipilih: {model_files[choice]}")

    # 2. Load model (deteksi otomatis in_channels)
    try:
        model, in_channels = load_model(model_path)
        print("✅ Model berhasil dimuat.\n")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return

    # 3. Minta path gambar RAW
    img_path = input("📂 Path gambar asli (RAW, format apa saja): ").strip().strip('"')
    if not os.path.exists(img_path):
        print("❌ File tidak ditemukan.")
        return

    # 4. Preprocessing sesuai tipe model
    print("\n⏳ Memproses gambar...")
    try:
        if in_channels == 1:
            print("   (Model single‑channel, hanya menggunakan gambar RAW)")
            input_tensor = preprocess_single(img_path)
        else:  # in_channels == 2
            print("   (Model hybrid, mengekstrak LSB visual secara otomatis)")
            input_tensor = preprocess_hybrid(img_path)
    except Exception as e:
        print(f"❌ Gagal memproses gambar: {e}")
        return

    # 5. Inferensi
    print("🧠 Melakukan inferensi...")
    probs = predict(model, input_tensor)
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    # 6. Tampilkan hasil
    print("\n" + "="*70)
    print("📊 HASIL PREDIKSI")
    print("="*70)
    print(f"Kelas terprediksi : {CLASSES[pred_class]} (index {pred_class})")
    print(f"Confidence        : {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nProbabilitas per kelas:")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 40)
        print(f"  {cls:6s}: {probs[i]:.4f}  {bar}")
    print("="*70)

if __name__ == "__main__":
    main()