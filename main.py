import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import shutil
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
CLASSES = ['CLEAN', 'LSB', 'PVD', 'BPCS']
MODELS_DIR = "./Inference/models"

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

    def forward(self, x):
        x = self.srm(x)
        x = self.layer1(x)
        x = self.type1_blocks(x)
        x = self.type2_blocks(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

def preprocess_single(image_path):
    """Untuk model single‑channel (RAW saja)."""
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = (np.array(img, dtype=np.uint8) / 127.5) - 1.0
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
    img_rgb = Image.open(image_path).convert("RGB")
    img_arr = np.array(img_rgb)
    red_lsb = (img_arr[:, :, 0] & 1) * 255
    green_lsb = (img_arr[:, :, 1] & 1) * 255
    blue_lsb = (img_arr[:, :, 2] & 1) * 255
    lsb_gray = (0.299 * red_lsb + 0.587 * green_lsb + 0.114 * blue_lsb).astype(np.uint8)
    reveal_img = Image.fromarray(lsb_gray).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    reveal_arr = np.array(reveal_img, dtype=np.uint8)
    reveal_norm = (reveal_arr / 127.5) - 1.0

    # 3. Gabungkan: (2, H, W)
    hybrid = np.stack([raw_norm, reveal_norm], axis=0)
    tensor = torch.from_numpy(hybrid).float().unsqueeze(0)  # (1, 2, H, W)
    return tensor.to(DEVICE)

def get_in_channels_from_checkpoint(checkpoint_path):
    """Membaca checkpoint dan mengembalikan jumlah input channel."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    weight = state_dict['layer1.0.weight']
    in_channels = weight.shape[1]
    return in_channels


class SteganographyDetectorApp:
    """Application class to show simple app design to do steganalysis"""
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detection App")
        self.root.minsize(600, 500)
        self.root.resizable(True, True) 

        self.display_max_size = (426, 240)
        self.current_image_path = None

        self.base_image_path = './temp/'
        self.setup_base_image_path(self.base_image_path)

        self.available_models = self.list_models()
        self.setup_ui()

    def setup_base_image_path(self, image_path):
        if not os.path.exists(image_path):
            os.mkdir(image_path)

    def list_models(self):
        if not os.path.exists(MODELS_DIR):
            return []
        return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')])

    def setup_ui(self):
        # Header
        header_label = tk.Label(self.root, text="Steganography Detection", font=("Helvetica", 18, "bold"), pady=10)
        header_label.pack()

        # Model Selection Frame
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)
        
        tk.Label(model_frame, text="Select Model:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        self.selected_model = tk.StringVar()
        if self.available_models:
            self.selected_model.set(self.available_models[0])
            self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model, values=self.available_models, width=40)
            self.model_dropdown.pack(side=tk.LEFT, padx=5)
        else:
            tk.Label(model_frame, text="No models found in Inference/models/", fg="red").pack(side=tk.LEFT)

        # Upload Button
        self.upload_btn = tk.Button(self.root, text="Select Image File", command=self.upload_image, bg="#e1e1e1", font=("Helvetica", 12), padx=20, pady=5)
        self.upload_btn.pack(pady=10)

        # Image Preview
        self.img_container = tk.LabelFrame(self.root, text="Image Preview", width=426, height=240)
        self.img_container.pack(pady=10)
        self.img_container.pack_propagate(False)

        self.display_label = tk.Label(self.img_container, text="No image loaded")
        self.display_label.pack(expand=True)

        # Prediction Result
        self.result_label = tk.Label(
            self.root, 
            text="Detection results will appear here...", 
            font=("Courier", 14),
            bg="#f0f0f0",
            fg="black",
            wraplength=550,
            relief=tk.SUNKEN,
            pady=15
        )
        self.result_label.pack(fill=tk.X, padx=20, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )

        if file_path:
            self.current_image_path = self.copy_base_image(file_path)
            self.load_and_display_image(self.current_image_path)
            self.run_steganalysis(self.current_image_path)

    def copy_base_image(self, file_path):
        target = os.path.join(self.base_image_path, 'base.png')
        shutil.copy2(file_path, target)
        return target

    def load_and_display_image(self, image_path):
        try:
            pil_image = Image.open(image_path)
            pil_image.thumbnail(self.display_max_size, Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.display_label.config(image=tk_image, text="") 
            self.display_label.image_ref = tk_image
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def run_steganalysis(self, image_path):
        if not self.selected_model.get():
            messagebox.showwarning("Warning", "Please select a model first.")
            return

        self.result_label.config(text="Running detection...", fg="blue")
        self.root.update_idletasks()

        model_path = os.path.join(MODELS_DIR, self.selected_model.get())
        
        try:
            in_channels = get_in_channels_from_checkpoint(model_path)
            model = SRNet(num_classes=len(CLASSES), in_channels=in_channels).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            
            # Handle possible class mismatch (if any model has different classes count)
            state_dict = checkpoint
            if state_dict['classifier.weight'].shape[0] != len(CLASSES):
                for key in list(state_dict.keys()):
                    if key.startswith('classifier'):
                        del state_dict[key]
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
            
            model.eval()

            if in_channels == 1:
                input_tensor = preprocess_single(image_path)
            else:
                input_tensor = preprocess_hybrid(image_path)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]

            self.result_label.config(text=f"Prediction: {CLASSES[pred_class]} ({confidence*100:.2f}%)", fg="black")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {e}")
            self.result_label.config(text="Inference failed.", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = SteganographyDetectorApp(root)
    root.mainloop()
