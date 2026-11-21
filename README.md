# Automated Brain Tumor MRI Segmentation with 2D U-Net

A PyTorch implementation of a custom **2D U-Net** encoder-decoder architecture with skip connections for pixel-level brain tumor segmentation on Lower-Grade Glioma (LGG) FLAIR MRI scans. Includes a Flask web interface that converts `.tif` medical slices into real-time binary masks and clinical overlays on standard CPU hardware.

![U-Net Segmentation Result](assets/segmentation_result.png)

## Interactive Flask Web Demo (CPU Inference)
The Flask web application (`app.py`) converts raw `.tif` MRI slices into browser-ready visualizations, computing the segmented tumor area percentage and rendering a red clinical overlay in real time on standard CPU hardware.

![Flask MRI Segmentation Demo](assets/web_demo.png)


## Architecture & Clinical Methodology
- **Custom 2D U-Net:** Built from scratch in PyTorch with progressive downsampling (`32 → 64 → 128 → 256` channels) and symmetric skip connections to preserve fine spatial tumor boundaries.
- **Dataset:** [LGG MRI Segmentation Dataset](https://kaggle.com) (`3,929` multi-channel MRI slices paired with manual expert segmentation masks).
- **Loss Function (`BCE + Dice Loss`):** Combines Binary Cross-Entropy with soft Dice Loss to overcome severe foreground-background pixel imbalance.

## Quantitative Evaluation (Test Split)

| Segmentation Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Mean Dice Similarity Coefficient** | **0.6899** | Strong spatial overlap across positive and negative slices |
| **Mean Intersection over Union (IoU)** | **0.6504** | Reliable tumor boundary delineation |

## Quickstart
