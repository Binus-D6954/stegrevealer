import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
CLASSES     = ['BPCS', 'CLEAN', 'LSB', 'PVD']

def get_srm_kernels():
    """
    30 SRM (Spatial Rich Model) high-pass filter kernels.
    Source: Fridrich & Kodovsky, IEEE TIFS 2012.
    Weights diset fixed (tidak ditraining).
    """
    # ── 5x5 kernels ──────────────────────────────────────────
    # Setiap kernel berukuran 5x5
    srm_kernels = np.zeros((30, 1, 5, 5), dtype=np.float32)

    # Kernel set 1: 1st order horizontal / vertical / diagonal
    # (edge detection residuals)
    k = 0
    
    # 1st order
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32)
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32)
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32)
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]], dtype=np.float32)
    srm_kernels[k] = f; k+=1

    # 2nd order
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 2
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]], dtype=np.float32) / 2
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,-2,0,0],[0,0,0,1,0],[0,0,0,0,0]], dtype=np.float32) / 2
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,-2,0,0],[0,1,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 2
    srm_kernels[k] = f; k+=1

    # 3rd order (horizontal)
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,3,-3,1],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 3
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,0,0,0],[0,0,0,0,0],[1,-3,3,-1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 3
    srm_kernels[k] = f; k+=1
    # 3rd order (vertical)
    f = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,1,0,0]], dtype=np.float32) / 3
    srm_kernels[k] = f; k+=1
    f = np.array([[0,0,1,0,0],[0,0,-3,0,0],[0,0,3,0,0],[0,0,-1,0,0],[0,0,0,0,0]], dtype=np.float32) / 3
    srm_kernels[k] = f; k+=1

    # 2D Laplacian-like
    f = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]], dtype=np.float32) / 4
    srm_kernels[k] = f; k+=1

    # Square kernels (2D 2nd order)
    f = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float32) / 4
    srm_kernels[k] = f; k+=1

    # 3x3 SRM embedded in 5x5
    base_3x3_list = [
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32) / 4,
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32) / 4,
        np.array([[-1,0,1],[0,0,0],[1,0,-1]], dtype=np.float32) / 2,
        np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32) / 4,
        np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.float32) / 4,
        np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32) / 2,
        np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float32) / 2,
        np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=np.float32) / 2,
        np.array([[-1,2,-2],[2,-4,2],[-1,2,-1]], dtype=np.float32) / 4,
        np.array([[2,-4,2],[-4,8,-4],[2,-4,2]], dtype=np.float32) / 8,
        np.array([[-1,2,-1],[0,0,0],[1,-2,1]], dtype=np.float32) / 2,
        np.array([[-1,0,1],[2,0,-2],[-1,0,1]], dtype=np.float32) / 2,
    ]
    for kern3 in base_3x3_list:
        if k >= 30:
            break
        f = np.zeros((5,5), dtype=np.float32)
        f[1:4, 1:4] = kern3
        srm_kernels[k] = f; k+=1

    # Fill remaining slots with rotations if needed
    while k < 30:
        base = srm_kernels[k % 14].copy()
        srm_kernels[k] = np.rot90(base, k % 4, axes=(1, 2))
        k += 1

    return torch.tensor(srm_kernels)

class SRMLayer(nn.Module):
    """Fixed SRM high-pass filter layer (tidak ditraining)"""
    def __init__(self):
        super().__init__()
        srm_weights = get_srm_kernels()  # (30, 1, 5, 5)
        self.register_buffer('weight', srm_weights)
        # Truncation activation (TLU) — clamp ke [-T, T]
        self.T = 3.0

    def forward(self, x):
        # x shape: (B, 1, H, W)
        out = nn.functional.conv2d(x, self.weight, padding=2)
        return out.clamp(-self.T, self.T)  # TLU activation


class Type1Block(nn.Module):
    """Residual block WITHOUT downsampling"""
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
    """Residual block WITH downsampling (AvgPool)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(3, stride=2, padding=1),
        )
        # Shortcut untuk menyamakan channel & spatial size
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
    """
    SRNet: Steganalysis Residual Network
    Boroumand et al., IEEE TIFS 2019
    
    Modified untuk multi-class (4 classes) steganalysis.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # Layer 0: Fixed SRM preprocessing
        self.srm = SRMLayer()  # output: (B, 30, H, W)

        # Layer 1: Conv to reduce to 64 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Type 1 Blocks (no downsampling)
        self.type1_blocks = nn.Sequential(
            Type1Block(64),
            Type1Block(64),
        )

        # Type 2 Blocks (with downsampling + channel expansion)
        self.type2_blocks = nn.Sequential(
            Type2Block(64,  128),  # 256 → 128
            Type2Block(128, 256),  # 128 → 64
            Type2Block(256, 512),  # 64  → 32
            Type2Block(512, 512),  # 32  → 16
            Type2Block(512, 512),  # 16  → 8
        )

        # Type 3: Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.srm(x)          # Fixed SRM filter
        x = self.layer1(x)       # Initial conv
        x = self.type1_blocks(x) # Type 1
        x = self.type2_blocks(x) # Type 2 (downsampling)
        x = self.gap(x)          # Global avg pool
        x = self.classifier(x)   # FC → logits
        return x


class RGBRevealer:
    @staticmethod
    def extract_lsb_rgb_visual(input_path, output_path):
        """Extract from simple image to RGB, saved into specific file path"""
        img = Image.open(input_path).convert("RGB")
        img_array = np.array(img)

        red_lsb   = (img_array[:, :, 0] & 1) * 255
        green_lsb = (img_array[:, :, 1] & 1) * 255
        blue_lsb  = (img_array[:, :, 2] & 1) * 255

        lsb_image = np.stack([red_lsb, green_lsb, blue_lsb], axis=2).astype(np.uint8)
        result = Image.fromarray(lsb_image)
        result.save(output_path)

class SteganographyDetectorApp:
    """Application class to show simple app design to do steganalysis"""
    def __init__(self, root):
        self.root = root
        self.root.title("Steganography Detection App")
        self.root.geometry("600x700")
        self.root.resizable(False, False) 

        self.display_max_size = (426, 240)
        self.current_image_path = None
        self.rgb_image_path = None

        self.base_image_path = './temp/'

        # Base image path setup to temporarily save image
        self.setup_base_image_path(self.base_image_path)

        self.setup_ui()

    def setup_base_image_path(self, image_path):
        """Create folder on specific image path if not exists."""
        if os.path.exists(image_path) and os.path.isdir(image_path):
            pass
        else:
            os.mkdir(image_path)

    def setup_ui(self):
        """Creates and packs the widgets onto the window."""

        header_label = tk.Label(
            self.root, 
            text="Upload an image to detect", 
            font=("Helvetica", 16, "bold"),
            pady=20
        )
        header_label.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(
            button_frame, 
            text="Select Image File", 
            command=self.upload_image,
            bg="#e1e1e1",
            font=("Helvetica", 12),
            padx=20, pady=5
        )
        self.upload_btn.pack()

        self.top_container = tk.LabelFrame(self.root, text="Main Image Preview", width=426, height=240)
        self.top_container.pack(pady=10)
        self.top_container.pack_propagate(False)

        self.top_display_label = tk.Label(self.top_container, text="No image loaded")
        self.top_display_label.pack(expand=True)

        self.bottom_container = tk.LabelFrame(self.root, text="RGB Image Preview", width=426, height=240)
        self.bottom_container.pack(pady=10)
        self.bottom_container.pack_propagate(False)

        self.bottom_display_label = tk.Label(self.bottom_container, text="No image loaded")
        self.bottom_display_label.pack(expand=True)

        self.result_label = tk.Label(
            self.root, 
            text="Detection results will appear here...", 
            font=("Courier", 14),
            bg="#f0f0f0",
            wraplength=550,
            relief=tk.SUNKEN,
            pady=20
        )
        self.result_label.pack(fill=tk.X, padx=20, pady=(0, 20))

    def upload_image(self):
        """Handles the file dialog and image loading process."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            # If the user selected a file (didn't click cancel)
            self.current_image_path = self.copy_base_image(file_path)
            self.rgb_image_path = self.base_image_path + 'rgb.png'
            RGBRevealer.extract_lsb_rgb_visual(self.current_image_path, self.rgb_image_path)
            self.load_and_display_image(self.current_image_path, self.rgb_image_path)
            # Run the prediction AFTER the image loads
            self.run_steganalysis(self.rgb_image_path)

    def copy_base_image(self, file_path):
        """Copy uploaded image to temporary saved image, for safety measures."""
        target = self.base_image_path + 'base.png'
        shutil.copy2(file_path, target)

        return target

    def load_and_display_image(self, normal_image_path, rgb_image_path):
        """Opens image using PIL, resizes it for display, and puts it in the Label."""
        try:
            normal_pil_image = Image.open(normal_image_path)
            rgb_pil_image = Image.open(rgb_image_path)

            normal_pil_image.thumbnail(self.display_max_size, Image.LANCZOS)
            rgb_pil_image.thumbnail(self.display_max_size, Image.LANCZOS)

            normal_tk_image = ImageTk.PhotoImage(normal_pil_image)
            rgb_tk_image = ImageTk.PhotoImage(rgb_pil_image)

            self.top_display_label.config(image=normal_tk_image, text="") 
            self.bottom_display_label.config(image=rgb_tk_image, text="")

            self.top_display_label.image_ref = normal_tk_image
            self.bottom_display_label.image_ref = rgb_tk_image

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.result_label.config(text="Error loading image.")

    def run_steganalysis(self, image_path):
        """Run steganalysis and renders the result to the UI."""
        self.result_label.config(text="Running algorithms to detect...", fg="blue")
        self.root.update_idletasks()

        checkpoint = torch.load("./srnet_best.pth", map_location=DEVICE)
        model = SRNet(num_classes=NUM_CLASSES).to(DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        img = Image.open(image_path).convert('RGB')
        img = val_test_transform(img)
        imgs = [img]
        tensor = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
        
        for _, idx, prob in zip([image_path], preds.cpu().numpy(), probs.cpu().numpy()):
            predicted = CLASSES[idx]
            confidence = prob[idx] * 100

            prediction_text = f"Prediction: {predicted} ({confidence:.1f}%)"

        self.result_label.config(text=prediction_text, fg="black")


if __name__ == "__main__":
    root = tk.Tk()
    app = SteganographyDetectorApp(root)
    root.mainloop()