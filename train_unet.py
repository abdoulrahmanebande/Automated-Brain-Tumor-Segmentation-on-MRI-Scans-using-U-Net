import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training U-Net on: {device}")

# 2. Custom MRI Dataset Loader (Pairs each MRI slice with its binary tumor mask)
class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, img_size=(128, 128)):
        self.img_size = img_size
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*_mask.tif"), recursive=True))
        self.img_paths = [p.replace("_mask.tif", ".tif") for p in self.mask_paths]
        print(f"Loaded {len(self.img_paths)} MRI slice-mask pairs.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(self.img_size)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize(self.img_size)

        img_tensor = TF.to_tensor(img)
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()  # Strict binary mask (0 or 1)
        return img_tensor, mask_tensor

# 3. U-Net Architecture (Encoder-Decoder with Skip Connections)
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

# 4. Combined BCE + Dice Loss & Clinical Segmentation Metrics
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice_score = (2. * intersection + smooth) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth)
        return bce_loss + (1 - dice_score.mean())

def compute_metrics(preds, targets):
    probs = (torch.sigmoid(preds) > 0.5).float()
    smooth = 1e-6
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2. * intersection + smooth) / (union + smooth)).mean().item()
    iou = ((intersection + smooth) / (union - intersection + smooth)).mean().item()
    return dice, iou

# 5. Prepare DataLoaders (80% Train / 20% Test)
dataset = BrainMRIDataset(os.path.join("data", "lgg-mri-segmentation", "kaggle_3m"))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = UNet().to(device)
criterion = BCEDiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 6. Train Loop (8 Epochs on 128x128 runs fast even on CPU)
epochs = 8
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss / len(train_dataset):.4f}")

# 7. Evaluate Dice & IoU + Generate Side-by-Side Comparison Image
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)
torch.save(model.state_dict(), "models/unet_brain_mri.pth")

model.eval()
dice_scores, iou_scores = [], []
sample_saved = False

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        d, i = compute_metrics(outputs, masks)
        dice_scores.append(d)
        iou_scores.append(i)

        # Save a visual comparison figure from a slice that contains a visible tumor
        if not sample_saved:
            for idx in range(imgs.size(0)):
                if masks[idx].sum() > 200:  # Pick a slice with a clear tumor
                    pred_mask = (torch.sigmoid(outputs[idx]) > 0.5).float().cpu().squeeze().numpy()
                    orig_img = imgs[idx].cpu().permute(1, 2, 0).numpy()
                    true_mask = masks[idx].cpu().squeeze().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(orig_img)
                    axes[0].set_title("Input Brain MRI")
                    axes[1].imshow(true_mask, cmap="gray")
                    axes[1].set_title("Ground Truth Tumor Mask")
                    axes[2].imshow(pred_mask, cmap="gray")
                    axes[2].set_title(f"U-Net Prediction (Dice: {d:.2f})")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig("assets/segmentation_result.png", dpi=150)
                    plt.close()
                    sample_saved = True
                    break

print(f"\n--- Test Set Segmentation Results ---")
print(f"Mean Dice Similarity Coefficient: {np.mean(dice_scores):.4f}")
print(f"Mean Intersection over Union (IoU): {np.mean(iou_scores):.4f}")
print("Saved visual comparison to assets/segmentation_result.png")
