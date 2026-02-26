# 🛰️ Satellite Ship Detection via Semantic Segmentation
### Airbus Kaggle Competition · U-Net from Scratch · Deep Learning · Remote Sensing

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square&logo=tensorflow)
![Kaggle](https://img.shields.io/badge/Kaggle-Airbus%20Ship%20Detection-20BEFF?style=flat-square&logo=kaggle)
![Architecture](https://img.shields.io/badge/Model-U--Net-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-red?style=flat-square)

---

## 🌍 Project Overview

Every day, thousands of ships cross the world's oceans — carrying cargo, fishing illegally,  
or evading maritime law. **Satellite imagery** can monitor all of them, but only if we can  
automatically detect ships pixel-by-pixel at scale.

This project builds a **U-Net deep learning model from scratch** to perform  
**semantic segmentation** on the Airbus Ship Detection Kaggle dataset —  
identifying the exact pixels in a satellite image that belong to a ship.

> **Task:** Given a 768×768 satellite image, predict a binary mask  
> where **1 = ship pixel** and **0 = ocean/background pixel**

---

## 📌 Image Classification vs. Object Detection vs. Semantic Segmentation

| Approach | What It Does | Output |
|---|---|---|
| **Classification** | Is there a ship? Yes / No | Single label |
| **Object Detection** | Where is the ship? | Bounding box |
| **Semantic Segmentation** | Which exact pixels are the ship? | Full pixel mask ← *this project* |

Semantic segmentation is the hardest and most precise of the three —  
and the most directly useful for real-world satellite monitoring systems.

---

## 📡 Dataset

| Property | Details |
|---|---|
| **Source** | [Airbus Ship Detection — Kaggle Competition](https://www.kaggle.com/competitions/airbus-ship-detection) |
| **Train images** | 768×768 RGB satellite images |
| **Labels** | `train_ship_segmentations_v2.csv` — Run-Length Encoded (RLE) masks |
| **Challenge** | Most images contain **no ships** — severe class imbalance |

> ⚠️ Dataset not included due to Kaggle licensing.  
> Download from the competition page and place in:  
> `../input/airbus-ship-detection/`

---

## 🧠 Where the Real Thinking Happened

### 1. Decoding Run-Length Encoding (RLE) — From Numbers to Pixels

Kaggle stores ship locations as **RLE strings** — a compressed format.  
Before any model can train, these must be decoded into actual pixel masks.

**Example RLE:** `56777 3` means pixels 56777, 56778, 56779 are ship pixels.

```python
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    ends = starts + lengths - 1
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi+1] = 1
    return img.reshape(shape).T   # Transpose — critical for correct orientation

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
```

**Key insight:** The `.T` transpose is non-obvious — without it, every mask  
is rotated 90° and ships appear in the wrong location. Caught and fixed during EDA.

---

### 2. Class Imbalance — The Hardest Problem in This Dataset

The dataset has a **catastrophic imbalance:**

```
Images with 0 ships  →  vast majority
Images with 1 ship   →  far fewer
Images with 7+ ships →  very rare
```

A naive model learns to predict "no ship" everywhere and still achieves high accuracy.  
The fix required a **two-step strategy:**

**Step 1 — Stratified train/test split:**
```python
train_ids, valid_ids = train_test_split(
    unique_img_ids,
    test_size=0.3,
    stratify=unique_img_ids['ships']   # preserve ship-count distribution
)
```

**Step 2 — Custom random undersampling across 7 ship-count bins:**
```python
def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].values[0] == 0:
        return in_df.sample(base_rep_val // 3)  # 500 samples for no-ship images
    else:
        return in_df.sample(base_rep_val)        # 1500 samples for ship images

# Group by ship count bin (0–7), apply undersampling per bin
balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
```

Grouping into **7 bins** (not just ship/no-ship) was the key decision —  
it ensured images with many ships weren't underrepresented relative to single-ship images.

---

### 3. U-Net Architecture — Built from Scratch

U-Net was chosen because it preserves **spatial resolution** through skip connections —  
critical for pixel-level ship boundary detection where exact shape matters.

```
Input (768×768×3)
      │
   [Encoder - Contracting Path]
   Conv2D(8)  → MaxPool  ─────────────────────────────────┐ skip
   Conv2D(16) → MaxPool  ──────────────────────────────┐  │ skip
   Conv2D(32) → MaxPool  ───────────────────────────┐  │  │ skip
   Conv2D(64) → MaxPool  ────────────────────────┐  │  │  │ skip
      │
   [Bottleneck]
   Conv2D(128) + GaussianNoise(0.1) + BatchNorm
      │
   [Decoder - Expanding Path]
   UpSampling2D ←─────────────────────────────── skip ┘
   UpSampling2D ←──────────────────────────────── skip ┘
   UpSampling2D ←───────────────────────────────── skip ┘
   UpSampling2D ←────────────────────────────────── skip ┘
      │
Output (768×768×1) — Binary ship mask
```

**Design choices:**
- `GaussianNoise(0.1)` in bottleneck — mitigates overfitting on imbalanced data
- `BatchNormalization` — stabilises gradients, allows higher learning rate
- `UpSampling2D` (not `Conv2DTranspose`) — simpler, avoids checkerboard artefacts

---

### 4. Loss Function — Why Accuracy Alone Fails Here

Standard binary cross-entropy fails on imbalanced segmentation tasks —  
the model minimises loss by predicting all-background.

**Solution: Combined Dice + BCE loss**

```python
def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, dtype=K.floatx())
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)

def dice_p_bce(y_true, y_pred, alpha=1e-3, beta=0.5):
    bce  = -beta * (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    dice = 1 - dice_coef(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dice
```

- **BCE** penalises wrong pixel predictions
- **Dice** directly optimises overlap between predicted and true mask
- Combined: model can't cheat by predicting all-background

---

### 5. Training Pipeline

```python
# Parameters
BATCH_SIZE    = 4
NB_EPOCHS     = 5
GAUSSIAN_NOISE = 0.1
MAX_TRAIN_STEPS = 200

# Callbacks
ModelCheckpoint   — saves best weights by val_dice_coef
ReduceLROnPlateau — halves LR after 3 epochs of no improvement
EarlyStopping     — stops after 15 epochs of no improvement

# Data augmentation
dg_args = dict(
    rotation_range   = 15,
    horizontal_flip  = True,
    vertical_flip    = True,
    data_format      = 'channels_last'
)
```

**Critical detail:** Image and mask generators must share the **same random seed**  
during augmentation — otherwise the image and its mask get different transformations  
and the model trains on misaligned data.

---

## 📊 Pipeline Summary

```
Raw Satellite Images (768×768 RGB)
           │
           ▼
  RLE Decoding → Binary Masks (768×768)
           │
           ▼
  EDA → Identify Class Imbalance
  ├── Stratified train/val split
  └── Undersample by 7 ship-count bins
           │
           ▼
  Data Augmentation
  ├── Rotation, H-flip, V-flip
  └── Shared seed for image-mask alignment
           │
           ▼
  U-Net (scratch) — Dice+BCE loss
  ├── GaussianNoise + BatchNorm
  └── UpSampling2D skip connections
           │
           ▼
  Pixel-level Ship Segmentation Masks
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **TensorFlow / Keras** | U-Net architecture, training, callbacks |
| **scikit-image** | Image I/O, RLE mask visualisation, montage |
| **Pandas / NumPy** | RLE decoding, data pipeline management |
| **Matplotlib / Seaborn** | Ship count distribution, training visualisations |
| **scikit-learn** | Stratified train/test split |

---

## 🚀 Real-World Relevance for Space Applications

| This Project | Space Industry Application |
|---|---|
| Pixel-level segmentation from satellite imagery | Spacecraft component detection from orbital images |
| RLE mask pipeline | Compressed telemetry data decoding |
| Class imbalance on rare objects | Rare anomaly detection in satellite sensor data |
| U-Net skip connections | Preserving spatial resolution in orbital image analysis |
| Combined Dice+BCE loss | Precision-recall tradeoff in mission-critical detection |

---

## 📁 Repository Structure

```
satellite-ship-detection-unet/
│
├── sdsi-notebook.ipynb     ← Full notebook
├── README.md               ← You are here
└── requirements.txt        ← Dependencies
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/satellite-ship-detection-unet
cd satellite-ship-detection-unet

# 2. Install dependencies
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn scikit-image

# 3. Download dataset from Kaggle
# kaggle competitions download -c airbus-ship-detection
# Place in: ../input/airbus-ship-detection/

# 4. Open the notebook
jupyter notebook sdsi-notebook.ipynb
```

---

## 🔭 Future Work

- Replace `UpSampling2D` with `Conv2DTranspose` for learnable upsampling
- Test **DeepLabV3+** or **Mask R-CNN** for instance-level ship segmentation
- Apply to **Sentinel-2 multispectral imagery** for enhanced ocean monitoring
- Add **confidence thresholding** on predicted masks for production deployment

---

## 📚 References

- Airbus Ship Detection Competition: https://www.kaggle.com/competitions/airbus-ship-detection
- U-Net Paper (Ronneberger et al., 2015): https://arxiv.org/abs/1505.04597
- Dice Loss Survey: https://arxiv.org/pdf/2006.14822.pdf
- TensorFlow Keras Docs: https://keras.io

---

## 👩‍💻 Author

**Mubeena Hussain**
MSc Statistics — University of Kerala
📧 mubeenahussain1205@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/mubeenahussain)
🐙 [GitHub](https://github.com/YOUR-USERNAME)

---

*"Every ship that tries to hide from the law is just a segmentation problem waiting to be solved."*
