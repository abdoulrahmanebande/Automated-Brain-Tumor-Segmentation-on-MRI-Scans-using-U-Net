import io
import base64
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

# 1. Define U-Net Architecture (Matches train_unet.py)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_conv(d1)

model = UNet()
model.load_state_dict(torch.load("models/unet_brain_mri.pth", map_location="cpu"))
model.eval()

def to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB").resize((128, 128))
        input_tensor = TF.to_tensor(img).unsqueeze(0)

        start = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            pred_mask = (torch.sigmoid(logits) > 0.5).float().squeeze().numpy()
        latency_ms = round((time.time() - start) * 1000, 2)

        tumor_percent = round(float(pred_mask.mean() * 100), 2)

        # Create visual binary mask image
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

        # Create red semi-transparent overlay on the original MRI
        orig_np = np.array(img, dtype=np.float32)
        overlay_np = orig_np.copy()
        overlay_np[pred_mask == 1] = overlay_np[pred_mask == 1] * 0.4 + np.array([255, 0, 0]) * 0.6
        overlay_img = Image.fromarray(overlay_np.astype(np.uint8))

        return render_template(
            "index.html",
            orig_img=to_base64(img),
            mask_img=to_base64(mask_img),
            overlay_img=to_base64(overlay_img),
            tumor_percent=tumor_percent,
            latency=latency_ms
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    