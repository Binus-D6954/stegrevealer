import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Konfigurasi
CLASSES = ['CLEAN', 'LSB', 'PVD', 'BPCS']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256

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

def get_in_channels(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    return state_dict['layer1.0.weight'].shape[1]

def preprocess_single(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.array(img, dtype=np.uint8) / 127.5) - 1.0
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

def preprocess_hybrid(image_path):
    raw_img = Image.open(image_path).convert('L')
    raw_img = raw_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    raw_norm = (np.array(raw_img, dtype=np.uint8) / 127.5) - 1.0
    img_rgb = Image.open(image_path).convert("RGB")
    img_arr = np.array(img_rgb)
    red_lsb = (img_arr[:, :, 0] & 1) * 255
    green_lsb = (img_arr[:, :, 1] & 1) * 255
    blue_lsb = (img_arr[:, :, 2] & 1) * 255
    lsb_gray = (0.299 * red_lsb + 0.587 * green_lsb + 0.114 * blue_lsb).astype(np.uint8)
    reveal_img = Image.fromarray(lsb_gray).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    reveal_norm = (np.array(reveal_img, dtype=np.uint8) / 127.5) - 1.0
    hybrid = np.stack([raw_norm, reveal_norm], axis=0)
    return torch.from_numpy(hybrid).float().unsqueeze(0).to(DEVICE)

def main():
    model_dir = "models"
    test_base_dir = r"../test"
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    categories = ['cover', 'stego_bpcs_01', 'stego_bpcs_04', 'stego_lsb_01', 'stego_lsb_04', 'stego_pvd_01', 'stego_pvd_04']

    all_results = {}

    for m_file in model_files:
        m_path = os.path.join(model_dir, m_file)
        in_channels = get_in_channels(m_path)
        model = SRNet(num_classes=len(CLASSES), in_channels=in_channels).to(DEVICE)
        model.load_state_dict(torch.load(m_path, map_location=DEVICE))
        model.eval()
        
        m_results = {}
        for cat in categories:
            cat_path = os.path.join(test_base_dir, cat)
            if not os.path.exists(cat_path): continue
            files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))][:10]
            
            correct = 0
            expected = "CLEAN" if "cover" in cat else ("LSB" if "lsb" in cat else ("PVD" if "pvd" in cat else "BPCS"))
            
            for f in files:
                i_path = os.path.join(cat_path, f)
                input_tensor = preprocess_single(i_path) if in_channels == 1 else preprocess_hybrid(i_path)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = CLASSES[np.argmax(probs)]
                    if pred_class == expected:
                        correct += 1
            
            m_results[cat] = f"{correct}/10"
        
        all_results[m_file] = m_results

    # Summary
    print("\n" + "="*110)
    print(f"{'Model Name':<35} | " + " | ".join([f"{c:<10}" for c in categories]))
    print("="*110)
    for m_file, res in all_results.items():
        print(f"{m_file:<35} | " + " | ".join([f"{res.get(c, 'N/A'):<10}" for c in categories]))
    print("="*110)

if __name__ == "__main__":
    main()
