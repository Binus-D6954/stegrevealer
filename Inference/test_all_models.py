import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import random

# =============================================================================
# Konfigurasi
# =============================================================================
MODEL_DIR = "models"
DATASET_BASE = r"..\Dataset\Dataset Split RAW BOS\test"
CLASSES_MAP = {
    'CLEAN': 'cover',
    'LSB': 'stego_lsb',
    'PVD': 'stego_pvd',
    'BPCS': 'stego_bpcs'
}
CLASSES_LIST = ['CLEAN', 'LSB', 'PVD', 'BPCS']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256

# =============================================================================
# Arsitektur SRNet
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

    def forward(self, x):
        x = self.srm(x)
        x = self.layer1(x)
        x = self.type1_blocks(x)
        x = self.type2_blocks(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

# =============================================================================
# Helper Functions
# =============================================================================
def get_in_channels_from_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    return state_dict['layer1.0.weight'].shape[1]

def extract_lsb_visual_array(input_path):
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    red_lsb   = (img_array[:, :, 0] & 1) * 255
    green_lsb = (img_array[:, :, 1] & 1) * 255
    blue_lsb  = (img_array[:, :, 2] & 1) * 255
    lsb_gray = (0.299 * red_lsb + 0.587 * green_lsb + 0.114 * blue_lsb).astype(np.uint8)
    return lsb_gray

def preprocess_image(image_path, in_channels):
    # RAW channel
    raw_img = Image.open(image_path).convert('L')
    raw_img = raw_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    raw_arr = np.array(raw_img, dtype=np.uint8)
    raw_norm = (raw_arr / 127.5) - 1.0
    
    if in_channels == 1:
        tensor = torch.from_numpy(raw_norm).float().unsqueeze(0).unsqueeze(0)
    else:
        reveal_arr = extract_lsb_visual_array(image_path)
        reveal_img = Image.fromarray(reveal_arr).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        reveal_arr = np.array(reveal_img, dtype=np.uint8)
        reveal_norm = (reveal_arr / 127.5) - 1.0
        hybrid = np.stack([raw_norm, reveal_norm], axis=0)
        tensor = torch.from_numpy(hybrid).float().unsqueeze(0)
    
    return tensor.to(DEVICE)

def test_model(model_name, payload_label):
    model_path = os.path.join(MODEL_DIR, model_name)
    in_channels = get_in_channels_from_checkpoint(model_path)
    model = SRNet(num_classes=4, in_channels=in_channels).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"\n" + "="*80)
    print(f"🚀 TESTING MODEL: {model_name} (Payload: {payload_label})")
    print(f"📌 Input Channels: {in_channels}")
    print("="*80)

    results = {}

    for cls_name, folder_prefix in CLASSES_MAP.items():
        if cls_name == 'CLEAN':
            folder_name = 'cover'
        else:
            folder_name = f"{folder_prefix}_{payload_label.replace('.', '')}"
        
        dir_path = os.path.join(DATASET_BASE, folder_name)
        if not os.path.exists(dir_path):
            print(f"⚠️  Folder {dir_path} tidak ditemukan.")
            continue
            
        images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < 25:
            print(f"⚠️  Hanya ada {len(images)} gambar di {folder_name}, mengambil semua.")
            sampled_images = images
        else:
            sampled_images = random.sample(images, 25)

        correct = 0
        total = len(sampled_images)
        
        print(f"\n📂 Testing {cls_name} ({folder_name}):")
        for img_name in sampled_images:
            img_path = os.path.join(dir_path, img_name)
            input_tensor = preprocess_image(img_path, in_channels)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                
                # Check accuracy based on CLASSES_LIST index
                actual_idx = CLASSES_LIST.index(cls_name)
                if pred_idx == actual_idx:
                    correct += 1
                
                print(f"   - {img_name:15s} | Pred: {CLASSES_LIST[pred_idx]:6s} | Conf: {probs[pred_idx]*100:6.2f}% | {'✅' if pred_idx == actual_idx else '❌'}")
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        results[cls_name] = accuracy
        print(f"📊 Accuracy for {cls_name}: {accuracy:.2f}% ({correct}/{total})")

    return results

def main():
    # Test Model P04
    results_p04_04 = test_model("srnet_best_p04.pth", "0.4")
    results_p04_01 = test_model("srnet_best_p04.pth", "0.1")
    
    # Test Model P01
    results_p01_04 = test_model("srnet_best_p01.pth", "0.4")
    results_p01_01 = test_model("srnet_best_p01.pth", "0.1")

    print("\n" + "="*80)
    print("📈 RINGKASAN AKHIR PENGUJIAN")
    print("="*80)
    
    def print_summary(model_name, payload, results):
        print(f"\n🔹 Model: {model_name} | Dataset: {payload}")
        for cls, acc in results.items():
            print(f"   {cls:6s}: {acc:6.2f}%")
        avg = sum(results.values()) / len(results)
        print(f"   OVERALL: {avg:6.2f}%")

    print_summary("srnet_best_p04.pth", "0.4", results_p04_04)
    print_summary("srnet_best_p04.pth", "0.1", results_p04_01)
    print_summary("srnet_best_p01.pth", "0.4", results_p01_04)
    print_summary("srnet_best_p01.pth", "0.1", results_p01_01)
    print("="*80)

if __name__ == "__main__":
    main()
