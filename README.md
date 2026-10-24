# MAAI901 — Vision Transformer for Skin Lesion Classification

**Student Name:** Kainaz Kaur Chohan
**Student ID:** 8273273  
**Course:** MAAI901 — Deep Learning for Visual Data

## Project Overview

This project investigates skin lesion classification using a pretrained Vision Transformer (ViT-B/16) on the HAM10000 dataset.

The research question is whether fully fine-tuning the pretrained ViT improves performance compared with training only its classification head.

## Dataset

HAM10000 contains 10,015 dermatoscopic images across seven diagnostic classes.

**Dataset source:** https://doi.org/10.7910/DVN/DBW86T

The dataset was split at lesion level to prevent the same lesion from appearing in multiple subsets.

- Training: 6,981 images
- Validation: 1,532 images
- Testing: 1,502 images

## Models and Training

Two configurations of `google/vit-base-patch16-224-in21k` were evaluated:

1. **Fully fine-tuned ViT:** All model parameters were trained.
2. **Frozen-backbone ViT:** Only the classification head was trained.

Both models were trained for 10 epochs using AdamW, class-weighted cross-entropy loss and the same dataset splits.

## Test Results

| Model | Accuracy | Macro-F1 | Melanoma Recall |
|---|---:|---:|---:|
| Majority-class reference | 66.8% | 0.114 | 0.000 |
| Frozen-backbone ViT | 59.8% | 0.489 | 0.593 |
| Fully fine-tuned ViT | **78.8%** | **0.702** | **0.725** |

Full fine-tuning improved classification performance, although 46 of 167 melanoma images were missed. The model has not been clinically validated and must not be used for medical diagnosis.

## Running the Notebook

1. Download HAM10000 from the dataset source above.
2. Create `MyDrive/HAM10000` in Google Drive.
3. Place `HAM10000_metadata.csv`, `HAM10000_images_part_1` and `HAM10000_images_part_2` inside that folder.
4. Open `MAAI901_8273273_HAM10000_ViT_Classification.ipynb` in Google Colab.
5. Select a GPU runtime and run the notebook cells in order.

The notebook was run using a T4 GPU and random seed 42. The dataset, image cache and model checkpoints are not included in this repository.

## Limitations

The dataset is highly imbalanced, skin-tone annotations are unavailable, and each model was trained using a single random seed. Further external validation would be required before any clinical application.

## References

Tschandl, P., Rosendahl, C. and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
